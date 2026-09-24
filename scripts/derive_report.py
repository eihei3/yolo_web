# derive_report.py - 读取 result 文件夹里的原始检测 CSV/JSON，计算有效派生指标并生成报告文档
# 输入: result/001_boxes.csv, result/001_summary.json
# 输出: result/001_数据分析报告.md
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "result"


def main(stem_arg=None):
    # 优先使用显式传入的文件名前缀（后端 import 调用时 sys.argv 是 uvicorn 参数，不可用）；
    # 其次取命令行参数，如 python derive_report.py 001；都没有则取最新结果
    if stem_arg:
        CSV_PATH = OUT_DIR / f"{stem_arg}_boxes.csv"
        if not CSV_PATH.exists():
            print(f"找不到 {CSV_PATH}")
            return
    elif len(sys.argv) > 1:
        CSV_PATH = OUT_DIR / f"{sys.argv[1]}_boxes.csv"
        if not CSV_PATH.exists():
            print(f"找不到 {CSV_PATH}")
            return
    else:
        candidates = sorted(OUT_DIR.glob("*_boxes.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("result/ 中没有检测结果，请先运行 mypredict.py")
            return
        CSV_PATH = candidates[0]
    stem = CSV_PATH.name[: -len("_boxes.csv")]
    JSON_PATH = OUT_DIR / f"{stem}_summary.json"
    REPORT_PATH = OUT_DIR / f"{stem}_数据分析报告.md"

    df = pd.read_csv(CSV_PATH)
    meta = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    W, H = meta["resolution"]
    fps = meta["fps"]
    duration = meta["duration_sec"]

    # ---------- 1. 每秒鱼量曲线 ----------
    df["sec"] = (df["time_sec"]).astype(int)
    # 以“每帧鱼数”为基础重算（有些帧可能 0 条鱼，CSV 里无记录，需要补齐全部帧）
    total_frames = meta["total_frames"]
    per_frame = df.groupby("frame").size().reindex(range(1, total_frames + 1), fill_value=0)
    frame_sec = (per_frame.index / fps).astype(int)
    per_sec = per_frame.groupby(frame_sec).mean().round(2)

    # ---------- 2. 九宫格区域分布（按检测框中心点）----------
    gx = np.clip((df["center_x"] / W * 3).astype(int), 0, 2)
    gy = np.clip((df["center_y"] / H * 3).astype(int), 0, 2)
    grid = np.zeros(9, dtype=int)
    for g in (gy * 3 + gx):
        grid[g] += 1
    grid_pct = (grid / grid.sum() * 100).round(1)

    # ---------- 3. 检测框尺寸（反映鱼在画面中的远近/大小）----------
    bw, bh = df["box_w"], df["box_h"]
    box_area = bw * bh
    frame_area = W * H
    size_ratio = (box_area / frame_area * 100)  # 单鱼框占画面百分比

    # ---------- 4. 鱼量波动（稳定性）----------
    counts = per_frame.values.astype(float)
    mean_cnt = counts.mean()
    std_cnt = counts.std()
    cv = std_cnt / mean_cnt if mean_cnt else 0          # 变异系数，越小越稳定
    # 显著波动：相邻帧鱼数变化 >= 3
    jumps = int((np.abs(np.diff(counts)) >= 3).sum())

    # ---------- 5. 置信度分层 ----------
    conf = df["confidence"]
    high = (conf >= 0.7).mean() * 100
    mid = ((conf >= 0.5) & (conf < 0.7)).mean() * 100
    low = (conf < 0.5).mean() * 100

    # ---------- 6. 高位/低位时段（按秒）----------
    peak_sec = int(per_sec.idxmax())
    low_sec = int(per_sec.idxmin())

    # 鱼量最高/最低的连续 5 秒窗口
    window = per_sec.rolling(5, min_periods=1).mean()
    busy_win = int(window.idxmax())
    quiet_win = int(window.idxmin())

    lines = []
    lines.append(f"# {meta['file']} 鱼类检测数据分析报告\n")
    lines.append(f"> 数据来源：YOLO 模型 `{meta['model']}` 对视频逐帧检测，"
                 f"共 {total_frames} 帧、{len(df)} 个真实检测框，置信度阈值 {meta['conf_threshold']}。\n")
    lines.append(f"> 视频：{duration}s / {fps}fps / {W}x{H}；检测耗时 {meta['processing_time_sec']}s。\n")

    lines.append("\n## 一、鱼量时间特征\n")
    lines.append(f"- 全程平均鱼量：**{mean_cnt:.2f} 条/帧**")
    lines.append(f"- 单帧峰值：**{meta['max_fish_in_frame']} 条**（{meta['max_at_sec']}s），"
                 f"谷值：**{meta['min_fish_in_frame']} 条**（{meta['min_at_sec']}s）")
    lines.append(f"- 鱼量变异系数：**{cv:.2f}**（反映数量波动大小，0 最稳定）")
    lines.append(f"- 相邻帧鱼数突变≥3条的次数：**{jumps} 次**")
    lines.append(f"- 鱼量最高秒：第 **{peak_sec}s**（{per_sec.max():.1f} 条）；"
                 f"最低秒：第 **{low_sec}s**（{per_sec.min():.1f} 条）")
    lines.append(f"- 最密集 5 秒窗口：**{busy_win}-{busy_win+4}s**（均值 {window.max():.1f} 条）；"
                 f"最稀疏 5 秒窗口：**{quiet_win}-{quiet_win+4}s**（均值 {window.min():.1f} 条）\n")

    lines.append("### 每秒平均鱼量\n")
    lines.append("| 秒 | 平均鱼量(条) | 秒 | 平均鱼量(条) |")
    lines.append("|---|---|---|---|")
    secs = list(per_sec.index)
    half = (len(secs) + 1) // 2
    for i in range(half):
        left = f"{secs[i]} | {per_sec.iloc[i]:.2f}"
        if i + half < len(secs):
            right = f"{secs[i+half]} | {per_sec.iloc[i+half]:.2f}"
        else:
            right = " | "
        lines.append(f"| {left} | {right} |")

    lines.append("\n## 二、鱼群空间分布\n")
    names = [["左上", "中上", "右上"], ["左中", "中央", "右中"], ["左下", "中下", "右下"]]
    hot_idx = int(grid_pct.argmax())
    cold_idx = int(grid_pct.argmin())
    flat_names = [n for row in names for n in row]
    lines.append("|   |   |   |")
    lines.append("|---|---|---|")
    for r in range(3):
        cells = []
        for c in range(3):
            idx = r * 3 + c
            mark = "**" if idx == hot_idx else ""
            cells.append(f"{mark}{names[r][c]} {grid_pct[idx]:.1f}%{mark}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"\n- 鱼最常出现：**{flat_names[hot_idx]}**（{grid_pct[hot_idx]:.1f}%）；"
                 f"最少出现：**{flat_names[cold_idx]}**（{grid_pct[cold_idx]:.1f}%）\n")

    lines.append("## 三、鱼体在画面中的尺寸\n")
    lines.append(f"- 检测框平均宽度 {bw.mean():.0f}px、高度 {bh.mean():.0f}px")
    lines.append(f"- 单鱼检测框平均占画面 **{size_ratio.mean():.2f}%**，"
                 f"最大 {size_ratio.max():.2f}%，最小 {size_ratio.min():.3f}%")
    lines.append("- 说明：框越大通常表示鱼离镜头越近；要换算真实体长需提供标尺。\n")

    lines.append("## 四、检测质量\n")
    lines.append(f"- 平均置信度：**{conf.mean():.3f}**")
    lines.append(f"- 高置信度(≥0.7)：**{high:.1f}%**；中等(0.5~0.7)：{mid:.1f}%；较低(<0.5)：{low:.1f}%")
    lines.append(f"- 有鱼画面占比：**{meta['fish_appearance_rate']*100:.1f}%**；检测框总数：{meta['total_detections']}\n")

    lines.append("## 五、口径与限制\n")
    lines.append("- 本报告全部由原始检测数据（框坐标、置信度）统计得到，**未使用跟踪**，"
                 "因此不含鱼的 ID、游速、运动轨迹、进出场计数。")
    lines.append("- 空间占比、尺寸占比为画面像素口径；物理单位（体长 cm、密度 条/m²、游速 cm/s）"
                 "需一次标尺标定后才能给出。")
    lines.append("- 数量为模型检测口径，受漏检/误检影响；置信度分层可作为结果可信度参考。\n")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已生成: {REPORT_PATH}")
    print(f"平均鱼量={mean_cnt:.2f} 变异系数={cv:.2f} 高置信占比={high:.1f}% "
          f"最热区域={flat_names[hot_idx]}({grid_pct[hot_idx]:.1f}%)")


if __name__ == "__main__":
    main()
