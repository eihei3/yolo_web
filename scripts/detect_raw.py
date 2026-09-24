# detect_raw.py - 自包含目标检测脚本（脱离大框架可独立运行）
# 用法:
#   python detect_raw.py                 检测 input/ 目录里的第一个视频/图片
#   python detect_raw.py <文件路径>      检测指定视频或图片
#   python detect_raw.py <路径> [帧数]   视频只处理前 N 帧（调试用，0=全部）
# 输出到 output/：xxx_boxes.csv、xxx_summary.json、xxx_annotated.mp4/.jpg
import sys
import os
import json
import csv
import time
from pathlib import Path

# —— 脚本位于 scripts/，全部路径相对项目根目录 ——
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["YOLO_CONFIG_DIR"] = str(PROJECT_ROOT / ".ultralytics_cfg")  # 配置写在包内，不污染系统目录

import cv2
import numpy as np
from ultralytics import YOLO
import ultralytics

MODEL_PATH = PROJECT_ROOT / "model" / "best.pt"
OUT_DIR = PROJECT_ROOT / "output"
INPUT_DIR = PROJECT_ROOT / "input"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONF = 0.3


def write_rows(csv_writer, boxes, frame_no, t_sec):
    """把一帧的检测框写入 CSV，返回本帧鱼数"""
    if boxes is None or len(boxes) == 0:
        return 0
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy()
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        csv_writer.writerow([frame_no, round(t_sec, 3),
                             round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1),
                             round(float((x1 + x2) / 2), 1), round(float((y1 + y2) / 2), 1),
                             round(float(x2 - x1), 1), round(float(y2 - y1), 1),
                             round(float(confs[i]), 4), int(clss[i])])
    return len(xyxy)


def detect_image(model, src: Path, out_dir: Path = None):
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    r = model.predict(source=str(src), conf=CONF, iou=0.7, save=False, verbose=False)[0]
    cv2.imwrite(str(out_dir / f"{src.stem}_annotated.jpg"), r.plot())

    csv_path = out_dir / f"{src.stem}_boxes.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_sec", "x1", "y1", "x2", "y2", "center_x", "center_y",
                    "box_w", "box_h", "confidence", "class"])
        n = write_rows(w, r.boxes, 1, 0.0)

    confs = r.boxes.conf.cpu().numpy() if n else np.array([])
    summary = {"file": src.name, "model": MODEL_PATH.name, "type": "image", "conf_threshold": CONF,
               "fish_count": int(n),
               "avg_confidence": round(float(confs.mean()), 4) if n else 0,
               "outputs": {"boxes_csv": csv_path.name, "annotated": f"{src.stem}_annotated.jpg"}}
    _save_summary(src.stem, summary, out_dir)
    print(f"图片检测完成：{n} 条鱼，平均置信度 {summary['avg_confidence']}")


def detect_video(model, src: Path, max_frames: int, out_dir: Path = None, progress_cb=None):
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_dir / f"{src.stem}_annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    csv_path = out_dir / f"{src.stem}_boxes.csv"
    counts, frames_with, total_boxes, conf_sum = [], 0, 0, 0.0
    max_cnt, max_t, min_cnt, min_t = -1, 0.0, None, 0.0
    frame_no = 0
    t0 = time.time()

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        cw = csv.writer(f)
        cw.writerow(["frame", "time_sec", "x1", "y1", "x2", "y2", "center_x", "center_y",
                     "box_w", "box_h", "confidence", "class"])
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_no >= max_frames):
                break
            frame_no += 1
            r = model.predict(source=frame, conf=CONF, iou=0.7, save=False, verbose=False)[0]
            writer.write(r.plot())
            n = write_rows(cw, r.boxes, frame_no, round(frame_no / fps, 3))
            counts.append(n)
            if n:
                frames_with += 1
                total_boxes += n
                conf_sum += float(r.boxes.conf.cpu().numpy().sum())
            if n > max_cnt:
                max_cnt, max_t = n, round(frame_no / fps, 2)
            if min_cnt is None or n < min_cnt:
                min_cnt, min_t = n, round(frame_no / fps, 2)
            if progress_cb:
                progress_cb(frame_no, total)
            if frame_no % 150 == 0:
                print(f"进度 {frame_no}/{total} ({frame_no/total*100:.0f}%)", flush=True)

    cap.release()
    writer.release()
    c = np.array(counts)
    summary = {"file": src.name, "model": MODEL_PATH.name, "type": "video",
               "conf_threshold": CONF,
               "fps": round(fps, 2), "total_frames": total, "processed_frames": frame_no, "resolution": [w, h],
               "duration_sec": round(total / fps, 2), "processing_time_sec": round(time.time() - t0, 1),
               "avg_fish_per_frame": round(float(c.mean()), 2),
               "max_fish_in_frame": int(max_cnt), "max_at_sec": max_t,
               "min_fish_in_frame": int(min_cnt if min_cnt is not None else 0), "min_at_sec": min_t,
               "frames_with_fish": frames_with,
               "fish_appearance_rate": round(frames_with / frame_no, 4) if frame_no else 0,
               "total_detections": total_boxes,
               "avg_confidence": round(conf_sum / total_boxes, 4) if total_boxes else 0,
               "outputs": {"boxes_csv": csv_path.name, "annotated": f"{src.stem}_annotated.mp4"}}
    _save_summary(src.stem, summary, out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _save_summary(stem, summary, out_dir: Path = None):
    out_dir = out_dir or OUT_DIR
    with open(out_dir / f"{stem}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    else:  # 未传路径则自动取 input/ 里第一个媒体文件
        medias = [p for p in INPUT_DIR.glob("*") if p.suffix.lower() in IMG_EXT or p.suffix in {".mp4", ".avi", ".mov", ".mkv"}]
        if not medias:
            print(f"未找到待检测文件：请把视频或图片放入 {INPUT_DIR}，或用 python detect_raw.py <文件路径>")
            return
        src = medias[0]
    if not src.exists():
        print(f"文件不存在: {src}")
        return

    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(f"使用模型: {MODEL_PATH}")
    print(f"使用 ultralytics 库: {Path(ultralytics.__file__).parent}")  # 证明加载的是包内副本
    model = YOLO(str(MODEL_PATH))
    if src.suffix.lower() in IMG_EXT:
        detect_image(model, src)
    else:
        detect_video(model, src, max_frames)


if __name__ == "__main__":
    main()
