import io
import sys
import uuid
import shutil
import zipfile
import threading
from pathlib import Path
from urllib.parse import quote

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))                  # 项目内 ultralytics 副本
sys.path.insert(0, str(BASE_DIR / "scripts"))      # 复用 detect_raw / derive_report

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import detect_raw
import derive_report

RESULT_DIR = BASE_DIR / "result"      # Web 检测产物目录（视频/CSV/JSON/报告）
INPUT_DIR = BASE_DIR / "input"        # 上传的原视频目录（按任务 ID 分子目录）
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")   # 允许上传的视频格式

app = FastAPI(title="鱼类检测系统 API")

jobs = {}                               # 任务记录表：job_id -> 任务状态（仅存内存）
run_lock = threading.Lock()             # 全局检测锁：同一时刻只允许一个检测任务


def disposition(filename: str) -> str:
    # ASCII 回退名 + RFC5987 UTF-8 名，兼容中文文件名
    return f"attachment; filename=download; filename*=UTF-8''{quote(filename)}"


def run_job(job_id: str, src: Path):
    """后台线程执行的检测任务：加载模型 → 逐帧检测 → 生成报告 → 更新任务状态"""
    job = jobs[job_id]
    try:
        from ultralytics import YOLO

        def progress_cb(frame_no, total):
            # 由检测流程逐帧回调，用于前端轮询展示进度
            job["frame"] = frame_no
            job["total"] = total
            job["progress"] = round(frame_no / total * 100, 1) if total else 0

        model = YOLO(str(detect_raw.MODEL_PATH))
        detect_raw.detect_video(model, src, 0, out_dir=RESULT_DIR, progress_cb=progress_cb)

        stem = src.stem                    # 视频文件名作为产物的前缀
        job["stem"] = stem
        try:                                # 通过脚本自动生成数据分析报告
            derive_report.main(stem)
        except Exception as e:
            print(f"报告生成失败: {e}")

        # 登记该任务产生的数据文件
        data_files = [RESULT_DIR / f"{stem}_boxes.csv",
                      RESULT_DIR / f"{stem}_summary.json",
                      RESULT_DIR / f"{stem}_数据分析报告.md"]
        job["data_files"] = [p.name for p in data_files if p.exists()]
        job["video_file"] = f"{stem}_annotated.mp4"
        if not (RESULT_DIR / job["video_file"]).exists():
            raise RuntimeError("检测视频生成失败")
        job["progress"] = 100
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        run_lock.release()                 # 无论成功与否都要释放检测锁


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    """接收上传视频：校验格式 → 抢占检测锁 → 保存文件 → 启动后台检测线程"""
    if not file.filename.lower().endswith(VIDEO_EXT):
        raise HTTPException(status_code=400, detail="请上传 mp4/avi/mov 视频文件")
    if not run_lock.acquire(blocking=False):      # 非阻塞抢锁，已有任务在跑则拒绝
        raise HTTPException(status_code=409, detail="已有检测任务正在进行，请稍候")

    job_id = uuid.uuid4().hex[:12]                # 生成 12 位任务 ID
    job_dir = INPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    src = job_dir / file.filename
    try:                                          # 分块（1MB）保存上传的视频
        with open(src, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
    except Exception:
        run_lock.release()
        raise HTTPException(status_code=500, detail="视频保存失败")

    jobs[job_id] = {"status": "processing", "progress": 0, "frame": 0, "total": 0}
    threading.Thread(target=run_job, args=(job_id, src), daemon=True).start()  # 异步检测
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    """查询任务进度/结果，前端定时轮询此接口"""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


def resolve_stem(job_id: str):
    """根据任务 ID 找到视频名前缀 stem：优先查内存任务记录；
    服务重启后记录丢失时，从 input/{job_id} 目录中的视频文件名推导"""
    job = jobs.get(job_id)
    if job is not None:
        if job["status"] != "done":
            raise HTTPException(status_code=400, detail="检测尚未完成")
        return job, job.get("stem")
    job_dir = _safe_child(INPUT_DIR, job_id)
    if not job_dir or not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="任务不存在")
    stems = [f.stem for f in job_dir.iterdir()
             if f.is_file() and f.suffix.lower() in VIDEO_EXT]
    return None, stems[0] if stems else None


@app.get("/api/download/video/{job_id}")
def download_video(job_id: str):
    """下载带检测框的标注视频"""
    _, stem = resolve_stem(job_id)
    path = RESULT_DIR / f"{stem}_annotated.mp4" if stem else None
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="检测视频不存在")
    return FileResponse(path, media_type="video/mp4",
                        headers={"Content-Disposition": disposition(path.name)})


DATA_SUFFIXES = ("_boxes.csv", "_summary.json", "_数据分析报告.md")


@app.get("/api/download/data/{job_id}")
def download_data(job_id: str):
    """把 CSV、JSON、分析报告打包成 ZIP 导出"""
    _, stem = resolve_stem(job_id)
    files = []
    if stem:                                      # 按固定后缀收集实际存在的产物
        for suffix in DATA_SUFFIXES:
            p = RESULT_DIR / f"{stem}{suffix}"
            if p.exists():
                files.append(p)
    if not files:
        raise HTTPException(status_code=404, detail="没有可导出的数据")

    buf = io.BytesIO()                            # 在内存中打包zip，不落临时文件
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, p.name)
    buf.seek(0)
    zip_name = f"{stem or 'result'}_检测数据.zip"
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": disposition(zip_name)})


def _safe_child(root: Path, name: str) -> Path | None:
    """拼接后校验路径仍在 root 目录内，防止路径穿越误删"""
    p = (root / name).resolve()
    try:
        p.relative_to(root.resolve())
    except ValueError:
        return None
    return p


RESULT_SUFFIXES = ("_annotated.mp4", "_boxes.csv", "_summary.json", "_数据分析报告.md")


@app.delete("/api/job/{job_id}")
def delete_job(job_id: str):
    """删除任务：同步清理上传的原视频目录和 result 中对应的全部检测产物"""
    job = jobs.get(job_id)
    if job is not None and job.get("status") == "processing":
        raise HTTPException(status_code=400, detail="检测进行中，暂不能删除")
    jobs.pop(job_id, None)                        # 移除内存任务记录

    deleted = []

    # 1) 删除上传的原视频目录 input/{job_id}/，并从其中的视频文件名推导结果文件名
    job_dir = _safe_child(INPUT_DIR, job_id)
    stems = set()
    if job_dir and job_dir.is_dir():
        stems = {f.stem for f in job_dir.iterdir() if f.is_file()}
        shutil.rmtree(job_dir, ignore_errors=True)
        deleted.append(f"input/{job_id}")

    # 2) 待删文件名：任务记录中的产物 + 重启后按原视频名推导出的产物
    names = set()
    if job is not None:
        if job.get("video_file"):
            names.add(job["video_file"])
        names.update(job.get("data_files", []))
    for stem in stems:
        for suffix in RESULT_SUFFIXES:
            names.add(f"{stem}{suffix}")

    if job is None and not deleted:              # 记录和目录都不存在才算找不到
        raise HTTPException(status_code=404, detail="任务不存在")

    for name in names:
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            continue                             # 拒绝含路径分隔符的文件名
        p = _safe_child(RESULT_DIR, name)
        if p and p.is_file():
            try:
                p.unlink()
                deleted.append(f"result/{name}")
            except OSError:
                pass                             # 文件被占用等情况跳过，不影响其余清理

    return {"ok": True, "deleted": deleted}


@app.get("/")
def index():
    return RedirectResponse(url="/static/index.html")   # 根路径跳转到上传页面


# 挂载静态文件目录，提供前端页面
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
