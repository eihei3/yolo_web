# mypredict.py - 正常检测视频，并把检测过程中得到的逐帧数据保存到 result 文件夹
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # 使用项目内的 ultralytics 副本

import cv2
from ultralytics import YOLO

MODEL_PATH = PROJECT_ROOT / "model" / "best.pt"
# 默认检测项目内的测试视频；也可命令行指定：python mypredict.py 路径/视频.mp4
SOURCE_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "test_video" / "video" / "NO.1" / "001.mp4"
RESULT_DIR = PROJECT_ROOT / "result"  # 检测数据输出到这个文件夹
BOX_FIELDS = ["frame", "time_sec", "x1", "y1", "x2", "y2", "center_x", "center_y",
              "box_w", "box_h", "confidence", "class"]


def write_rows(writer, boxes, frame_no, t_sec):
    """把一帧里的每个检测框写成一行，返回本帧鱼数。"""
    if boxes is None or len(boxes) == 0:
        return 0
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy()
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        writer.writerow([frame_no, round(t_sec, 3),
                         round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1),
                         round(float((x1 + x2) / 2), 1), round(float((y1 + y2) / 2), 1),
                         round(float(x2 - x1), 1), round(float(y2 - y1), 1),
                         round(float(confs[i]), 4), int(clss[i])])
    return len(xyxy)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取视频基础信息（帧率、总帧数、分辨率）
    cap = cv2.VideoCapture(str(SOURCE_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # —— 和原来完全一样的检测方式 ——
    model = YOLO(str(MODEL_PATH))
    t0 = time.time()
    results = model.predict(
        source=str(SOURCE_PATH),  # 预测的目标
        # conf=0.5,  # 只显示置信度大于0.5的目标
        save=True,  # 是否保存
        show=False,  # 是否直接展示
        save_txt=True,
    )

    # 把逐帧、逐检测框的数据写入 result 文件夹
    stem = SOURCE_PATH.stem
    csv_path = RESULT_DIR / f"{stem}_boxes.csv"
    json_path = RESULT_DIR / f"{stem}_summary.json"

    counts, frames_with_fish, total_detections, conf_sum = [], 0, 0, 0.0
    max_cnt, max_t, min_cnt, min_t = -1, 0.0, None, 0.0
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(BOX_FIELDS)
        for frame_no, r in enumerate(results, start=1):
            t_sec = frame_no / fps
            n = write_rows(writer, r.boxes, frame_no, t_sec)
            counts.append(n)
            if n:
                frames_with_fish += 1
                total_detections += n
                conf_sum += float(r.boxes.conf.cpu().numpy().sum())
            if n > max_cnt:
                max_cnt, max_t = n, t_sec
            if min_cnt is None or n < min_cnt:
                min_cnt, min_t = n, t_sec

    processed = len(results)
    summary = {
        "file": SOURCE_PATH.name,
        "model": MODEL_PATH.name,
        "type": "video",
        "conf_threshold": 0.25,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "processed_frames": processed,
        "resolution": [width, height],
        "duration_sec": round(total_frames / fps, 2),
        "processing_time_sec": round(time.time() - t0, 1),
        "avg_fish_per_frame": round(sum(counts) / processed, 2) if processed else 0,
        "max_fish_in_frame": int(max_cnt),
        "max_at_sec": round(max_t, 2),
        "min_fish_in_frame": int(min_cnt if min_cnt is not None else 0),
        "min_at_sec": round(min_t, 2),
        "frames_with_fish": frames_with_fish,
        "fish_appearance_rate": round(frames_with_fish / processed, 4) if processed else 0,
        "total_detections": total_detections,
        "avg_confidence": round(conf_sum / total_detections, 4) if total_detections else 0,
        "outputs": {"boxes_csv": csv_path.name, "annotated": ""},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"检测数据已保存到文件夹: {RESULT_DIR}")


if __name__ == "__main__":
    main()
