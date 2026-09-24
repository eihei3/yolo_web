# 基于鱼群计数的目标检测系统（YOLO\_web）

一个面向水下鱼类视频的 Web 检测系统：在浏览器上传视频，后端使用 YOLO 模型逐帧检测，实时显示检测进度，完成后可导出带标注框的检测视频、逐帧检测数据（CSV/JSON）以及自动生成的数据分析报告（Markdown）。

## 功能特性

* 浏览器上传视频（支持点击选择与拖拽），兼容 mp4 / avi / mov / mkv
* YOLO 模型逐帧检测，页面实时显示"当前帧 / 总帧数 / 百分比"进度
* 检测完成后一键导出：标注视频、逐帧检测框数据、汇总信息、数据分析报告（以ZIP文件导出）
* 移除文件时同步删除该任务上传的原视频与全部检测产物（⚠行为不可逆）
* 同一时刻仅运行一个检测任务，避免抢占算力
* 数据分析报告自动生成，包含鱼量时间特征、空间分布九宫格、鱼体尺寸、检测质量等统计

## 目录结构

```
YOLO\_web/
├── app.py                            # FastAPI 后端（上传/检测/进度/导出/删除接口）
├── openHTML.bat                # Windows 一键启动脚本（启动服务并打开浏览器）
├── requirements.txt             # Python 依赖清单
├── model/
│   └── best.pt                     # YOLO 检测模型权重（必须保留）
├── static/
│   └── index.html               # 前端单页（上传、进度、导出、删除）
├── scripts/
│   ├── detect\_raw.py          # 核心检测流程：逐帧推理、写 CSV、生成标注视频
│   └── derive\_report.py      # 读取 CSV/JSON，生成数据分析报告 md
├── ultralytics/                   # 项目内置的 ultralytics 库副本（无需 pip 安装）
├── input/                         # 上传的原视频（按任务 ID 分目录，删除任务时清理）
├── result/                        # Web 检测的全部产物（视频/CSV/JSON/报告）
└── output/                      # 命令行直接跑 detect\_raw.py 时的输出目录
```

## 环境要求

* **操作系统**：Windows 10 / 11
* **Python**：3.10 及以上（Python 3.13 实测可运行）
* **硬件**：普通 CPU 即可运行（安装 CPU 版 PyTorch）；有 NVIDIA 显卡时安装 CUDA 版 PyTorch 可显著加快检测
* **浏览器**：Edge、Chrome 等现代浏览器
* 磁盘空间与所处理视频大小相当（用于保存原视频与标注视频）

## 安装依赖

建议使用虚拟环境（venv 或 conda），然后在项目根目录安装：

```bash
pip install -r requirements.txt
```

注意：**ultralytics已内置在项目文件夹中，不需要执行 `pip install ultralytics`**，代码会自动优先加载项目内的副本。

PyTorch 的安装说明：

* **CPU 版本**（无 NVIDIA 显卡，或不确定时选这个）：直接执行上面的 `pip install -r requirements.txt` 即可，pip 会自动安装 CPU 版 torch。
* **GPU 版本**（有 NVIDIA 显卡）：先按 PyTorch 官网选择与本机 CUDA 对应的命令安装 torch/torchvision，再安装其余依赖，例如 CUDA 12.1：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

可用下面命令确认 GPU 是否生效，输出 `True` 即表示走显卡：

```bash
python -c "import torch; print(torch.cuda.is\_available())"
```

主要依赖库说明：`fastapi` + `uvicorn` + `python-multipart` 提供 Web 接口与文件上传；`ultralytics`（内置）与 `torch` 提供模型推理；`opencv-python` 负责视频读写；`numpy`、`pandas` 负责数据统计；`matplotlib`、`seaborn`、`scipy`、`pillow`、`tqdm`、`psutil`、`pyyaml`、`requests`、`py-cpuinfo`、`ultralytics-thop` 为 ultralytics 运行所需。

## 运行方式

### 方式一：脚本启动（脚本为 Windows 批处理；Linux/macOS 可直接用命令行方式启动）

双击项目根目录的 **`openHTML.bat`**：脚本会自动切换到项目目录、启动后端服务并打开默认浏览器访问页面。

* 使用期间请保持弹出的黑色命令行窗口不要关闭，关闭即停止服务
* 若浏览器打开时页面未就绪，刷新一次即可
* 默认地址为 http://localhost:8000/static/index.html

### 方式二：命令行启动

在项目根目录执行：

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

启动后在浏览器手动打开 http://127.0.0.1:8000/ 。

修改 `app.py` 或 `scripts/` 下的 Python 代码后，需要重启服务（在命令行窗口按 `Ctrl+C` 后重新启动；双击 bat 的用户关闭窗口后重新双击）。

### 使用流程

1. 在页面中点击或拖拽上传一个视频文件（上传后仅显示文件名，不在页面内嵌播放）
2. 点击"检测视频"，等待进度条到 100%（检测期间不能开始第二个任务）
3. 完成后点击"导出检测视频"下载标注视频，或点击"导出数据"下载包含 CSV/JSON/报告的 ZIP
4. 点击"移除文件"并确认，会同步删除该任务的原视频与全部检测产物；检测进行中无法删除

## 接口文档

服务基址：`http://127.0.0.1:8000`。所有接口均为 JSON 交互（文件上传除外）。

### 1\. 上传视频并开始检测

`POST /api/detect`

* 请求体：`multipart/form-data`，字段名 `file`，文件扩展名为 mp4 / avi / mov / mkv
* 行为：保存视频后立即返回任务 ID，检测在后台线程异步执行
* 成功响应 `200`：

```json
{ "job\_id": "5c52cfef6146" }
```

* 错误响应：`400` 文件类型不支持；`409` 已有检测任务正在进行；`500` 视频保存失败

### 2\. 查询检测进度

`GET /api/status/{job\_id}`

* 路径参数：`job\_id` 为上传时返回的任务 ID
* 检测中响应：

```json
{ "status": "processing", "progress": 42.3, "frame": 570, "total": 1349 }
```

* 完成响应：

```json
{ "status": "done", "progress": 100, "frame": 1349, "total": 1349,
  "stem": "001", "video\_file": "001\_annotated.mp4",
  "data\_files": \["001\_boxes.csv", "001\_summary.json", "001\_数据分析报告.md"] }
```

* 失败响应中 `status` 为 `"error"` 并带 `error` 字段；任务 ID 不存在返回 `404`
* 前端按 1.5 秒间隔轮询该接口

### 3\. 下载检测视频

`GET /api/download/video/{job\_id}`

* 成功返回标注视频文件流，`Content-Type: video/mp4`，响应头携带中文文件名（RFC5987 编码）
* `400` 检测尚未完成；`404` 任务或视频不存在
* 说明：即使服务重启过（内存任务记录丢失），只要 `input/{job\_id}/` 目录还在，接口仍能找到对应结果

### 4\. 导出检测数据（ZIP）

`GET /api/download/data/{job\_id}`

* 成功返回 `application/zip` 压缩包，命名为 `{视频名}\_检测数据.zip`，内含实际存在的文件：

  * `{名}\_boxes.csv`：逐帧检测框明细
  * `{名}\_summary.json`：检测汇总信息
  * `{名}\_数据分析报告.md`：自动生成的分析报告
* `400` 检测尚未完成；`404` 没有可导出的数据

### 5\. 删除任务及全部文件

`DELETE /api/job/{job\_id}`

* 行为：删除 `input/{job\_id}/` 整个目录（上传的原视频），并删除 result 目录下该视频的标注视频、CSV、JSON、分析报告
* 服务重启后内存记录丢失时，会根据上传目录中的视频文件名自动推导出结果文件并清理
* 成功响应：

```json
{ "ok": true,
  "deleted": \["input/5c52cfef6146", "result/001\_annotated.mp4",
              "result/001\_boxes.csv", "result/001\_summary.json",
              "result/001\_数据分析报告.md"] }
```

* `400` 检测进行中拒绝删除；`404` 任务记录与上传目录均不存在
* 接口对文件名做了路径穿越校验，不可能删除 input/result 目录之外的文件

### 6\. 页面与静态资源

* `GET /`：重定向到上传页面
* `GET /static/index.html`：前端页面

交互式接口文档（Swagger UI）也可直接访问 http://127.0.0.1:8000/docs 在线调试。

## 输出数据说明

逐帧明细 `\*\_boxes.csv` 每一行代表一帧中的一条鱼（一帧多条鱼则多行），字段为：

* `frame`：帧序号（从 1 开始）
* `time\_sec`：该帧对应的视频时间（秒）
* `x1, y1, x2, y2`：检测框左上、右下角像素坐标
* `center\_x, center\_y`：检测框中心点像素坐标
* `box\_w, box\_h`：检测框宽高（像素）
* `confidence`：检测置信度（阈值 0.3）
* `class`：类别编号

`\*\_summary.json` 记录视频分辨率、帧率、时长、总帧数、检测框总数、最大/最小单帧鱼数及出现时间、有鱼画面占比、检测耗时等汇总信息。

`\*\_数据分析报告.md` 由 `scripts/derive\_report.py` 生成，含五个部分：鱼量时间特征（每秒鱼量、峰值谷值、波动统计）、鱼群空间分布（九宫格占比）、鱼体在画面中的尺寸、检测质量（置信度分层）、统计口径与限制。

## 脚本的独立用法

不启动 Web 服务时，也可在 `scripts/` 目录直接运行核心脚本，产物输出到 `output/`：

```bash
# 检测 input/ 目录里的第一个视频或图片
python scripts/detect\_raw.py

# 检测指定文件
python scripts/detect\_raw.py 路径/视频.mp4

# 只处理前 100 帧（调试用，0 表示全部）
python scripts/detect\_raw.py 路径/视频.mp4 100

# 为最新一次检测结果生成/重新生成分析报告（无参数取最新 CSV）
python scripts/derive\_report.py

# 为指定文件名前缀生成报告，如 001\_boxes.csv → 001\_数据分析报告.md
python scripts/derive\_report.py 001
```

## 常见问题

* **页面打不开 / 提示连接失败**：确认命令行窗口仍在运行；端口被占用时关掉占用 8000 端口的旧进程，或修改启动命令中的 `--port`。
* **检测很慢**：CPU 推理属于正常现象；有 NVIDIA 显卡请改装 CUDA 版 PyTorch。
* **同一时间只能检测一个视频**：这是设计限制（单卡串行），第二个上传请求会收到 `409`，待当前任务结束后再试。
* **重启服务后任务状态丢失**：任务进度记录保存在内存中，重启后无法继续轮询，但已生成的文件仍在磁盘；下载和删除接口会根据 `input/{job\_id}/` 目录自动找回对应结果。

