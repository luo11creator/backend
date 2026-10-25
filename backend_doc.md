# 多维盾卫 — 后端API完整接入指南

> 写给后端开发新手（学妹）的零基础教程

---

## 目录

1. [架构总览](#1-架构总览)
2. [环境准备](#2-环境准备)
3. [启动后端服务](#3-启动后端服务)
4. [API接口详解](#4-api接口详解)
5. [Android端如何调用](#5-android端如何调用)
6. [从模拟到真实AI模型](#6-从模拟到真实ai模型)
7. [调试技巧](#7-调试技巧)
8. [常见问题](#8-常见问题)

---

## 1. 架构总览

```
┌─────────────────┐     HTTP/HTTPS      ┌──────────────────┐     Python调用      ┌──────────────┐
│   Android App    │ ◄────────────────► │  FastAPI 后端     │ ◄───────────────► │  AI模型推理   │
│  (Java/Kotlin)   │   JSON / 文件上传   │  (Python 3.8+)   │                    │  PyTorch/TF   │
└─────────────────┘                     └──────────────────┘                    └──────────────┘
     │                                        │
     │  用户操作:                               │  端口: 8000
     │  - 录音 / 选视频 / 输入文本               │  提供 REST API
     │  - 上传文件                              │  接收文件 → 调用模型 → 返回结果
     │  - 显示检测结果                           │
```

**数据流**（以视频检测为例）：
1. 用户在App中选择视频 → App读取文件
2. App通过HTTP POST上传到 `http://你的服务器:8000/api/v1/detect/video`
3. 后端收到文件 → 保存临时文件 → 调用AI模型推理
4. 后端返回JSON结果（风险评分、风险等级、详细分析）
5. App解析JSON → 跳转到结果页面展示

---

## 2. 环境准备

### 2.1 安装Python依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 安装依赖
pip install fastapi uvicorn python-multipart
```

就这三个包，不需要装PyTorch（模型推理阶段再加）。

### 2.2 验证安装

```bash
python -c "import fastapi; print(fastapi.__version__)"
# 输出: 0.100.0 或更高版本
```

---

## 3. 启动后端服务

### 3.1 首次启动

```bash
cd backend
python server.py
```

看到以下输出表示成功：
```
==============================================================
  多维盾卫 - 后端AI检测服务
  API文档: http://localhost:8000/docs
==============================================================
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3.2 验证服务

浏览器打开 `http://localhost:8000/`，看到：
```json
{
  "service": "多维盾卫 AI检测服务",
  "version": "1.0.0",
  "status": "running"
}
```

### 3.3 交互式API文档（推荐）

打开 `http://localhost:8000/docs`，这是FastAPI自动生成的Swagger文档。

**你可以直接在网页上测试API** — 点击任意接口 → "Try it out" → 填入参数 → Execute。不需要写任何代码就能测试。

### 3.4 让手机能访问到后端

手机和电脑需要在**同一WiFi**下：

```bash
# 查看电脑IP地址
# Windows:
ipconfig
# 找 "无线局域网适配器 WLAN" 下的 IPv4 地址，比如 192.168.1.105

# Mac/Linux:
ifconfig | grep "inet "
```

然后修改 `NetworkConfig.java` 中的 `BASE_URL`：
```java
// 开发环境 — 指向你的电脑IP
public static final String BASE_URL = "http://192.168.1.105:8000";
```

**注意**: 用的是 `http://` 不是 `https://`，因为本地开发不需要SSL证书。

---

## 4. API接口详解

### 4.1 接口总览

| 方法 | 路径 | 用途 | 请求格式 |
|------|------|------|---------|
| GET | `/` | 健康检查 | - |
| POST | `/api/v1/detect/audio` | 音频检测 | multipart/form-data |
| POST | `/api/v1/detect/video` | 视频检测 | multipart/form-data |
| POST | `/api/v1/detect/text` | 文本分析 | application/json |
| GET | `/api/v1/result/{task_id}` | 查询结果 | - |

### 4.2 音频检测接口

**请求**:
```
POST /api/v1/detect/audio
Content-Type: multipart/form-data

参数:
  audio: 音频文件  (必填, .wav .mp3 .m4a .aac .flac .ogg)
  model: 模型选择  (可选, 默认 "aasist", 可选 "rawnet2")
```

**用curl测试**:
```bash
curl -X POST "http://localhost:8000/api/v1/detect/audio" \
  -F "audio=@test_audio.wav" \
  -F "model=aasist"
```

**返回示例**:
```json
{
  "task_id": "task_a1b2c3d4e5f6",
  "status": "completed",
  "risk_score": 0.1234,
  "risk_level": 1,
  "verdict": "真实",
  "detection_type": "audio",
  "model_used": "aasist (mock)",
  "details": {
    "aasist_score": 0.0863,
    "rawnet2_score": 0.0371,
    "sample_rate": 16000,
    "segments_analyzed": 5
  },
  "suggestions": [
    "未发现明显伪造痕迹",
    "但仍建议保持基本安全防范意识"
  ],
  "processed_at": "2026-07-31T20:30:00"
}
```

**risk_score 含义**:
| 数值范围 | risk_level | verdict | 含义 |
|---------|-----------|---------|------|
| 0.00 - 0.19 | 1 | 真实 | 基本确定是真人 |
| 0.20 - 0.39 | 2 | 低风险 | 有轻微异常 |
| 0.40 - 0.59 | 3 | 可疑 | 需要人工复核 |
| 0.60 - 0.79 | 4 | 高风险 | 很大概率是AI伪造 |
| 0.80 - 1.00 | 5 | 伪造 | 基本确定是AI生成 |

### 4.3 视频检测接口

**请求**:
```
POST /api/v1/detect/video
Content-Type: multipart/form-data

参数:
  video: 视频文件      (必填, .mp4 .mov .avi .mkv .webm)
  model: 模型选择      (可选, 默认 "efficientnet-b4")
  frame_rate: 采样帧率 (可选, 默认 5, 每秒抽几帧)
```

**用curl测试**:
```bash
curl -X POST "http://localhost:8000/api/v1/detect/video" \
  -F "video=@test_video.mp4" \
  -F "model=efficientnet-b4" \
  -F "frame_rate=5"
```

**返回格式与音频相同**，`detection_type` 为 `"video"`，`details` 中包含帧级别分析数据。

### 4.4 文本分析接口

**请求**:
```
POST /api/v1/detect/text
Content-Type: application/json

{
  "text": "您好，我是市公安局的，你的身份证涉嫌洗钱案件...",
  "model": "bert-base-chinese"
}
```

**用curl测试**:
```bash
curl -X POST "http://localhost:8000/api/v1/detect/text" \
  -H "Content-Type: application/json" \
  -d '{"text":"您好我是市公安局的，你的身份证涉嫌洗钱案件，需要配合调查。请立即将资金转入安全账户。","model":"bert-base-chinese"}'
```

**返回中额外包含**：
```json
{
  "fraud_type": "冒充公检法",
  "details": {
    "fraud_keywords_found": 5,
    "urgent_keywords_found": 2,
    "money_keywords_found": 3,
    "possible_types": ["冒充公检法", "冒充领导/熟人"]
  }
}
```

---

## 5. Android端如何调用

### 5.1 Android网络层架构

```
Fragment (UI层)
    │
    ├─ 用户点击"开始检测"
    ├─ 调用 ApiService.getInstance().detectVideo(file, callback)
    │
    ▼
ApiService (网络层)
    │
    ├─ 构建 OkHttp Multipart请求
    ├─ POST 到 NetworkConfig.BASE_URL + endpoint
    ├─ 异步回调 → 主线程更新UI
    │
    ▼
后端 FastAPI
    │
    ├─ 接收文件 → 调用AI模型 → 返回JSON
```

### 5.2 代码调用示例

假设你在 `VideoDetectFragment` 中，用户选好了视频文件，想发送到后端检测：

```java
import com.example.deepfake_fight.service.ApiService;
import com.example.deepfake_fight.service.NetworkConfig;

// 1. 获取API服务实例（单例）
ApiService api = ApiService.getInstance();

// 2. 将Uri转为File
// (你的 selectedVideoUri 需要先通过 ContentResolver 拷贝到缓存目录)
File videoFile = copyUriToCache(selectedVideoUri);

// 3. 调用检测
api.detectVideo(videoFile, new ApiService.DetectCallback() {

    @Override
    public void onProgress(String status) {
        // 在UI线程中执行 — 更新进度文字
        binding.tvAnalyzeProgress.setText(status);
    }

    @Override
    public void onSuccess(JSONObject result) {
        // 在UI线程中执行 — 解析结果
        float riskScore = (float) result.getDouble("risk_score");
        int riskLevel = result.getInt("risk_level");
        String verdict = result.getString("verdict");

        // 跳转到结果页
        Bundle args = new Bundle();
        args.putFloat("risk_score", riskScore);
        args.putInt("risk_level", riskLevel);
        args.putString("verdict", verdict);
        Navigation.findNavController(requireView())
                 .navigate(R.id.nav_result, args);
    }

    @Override
    public void onError(String errorMessage) {
        // 在UI线程中执行 — 显示错误
        Toast.makeText(getContext(), "检测失败: " + errorMessage,
                       Toast.LENGTH_LONG).show();
    }
});
```

### 5.3 Uri转File工具方法

```java
// 在Fragment中添加此方法
private File copyUriToCache(Uri uri) throws IOException {
    String fileName = "video_" + System.currentTimeMillis() + ".mp4";
    File destFile = new File(requireContext().getCacheDir(), fileName);

    try (InputStream in = requireContext().getContentResolver().openInputStream(uri);
         OutputStream out = new FileOutputStream(destFile)) {
        byte[] buf = new byte[8192];
        int len;
        while ((len = in.read(buf)) > 0) {
            out.write(buf, 0, len);
        }
    }
    return destFile;
}
```

### 5.4 网络配置

AndroidManifest.xml 中已添加：
```xml
<uses-permission android:name="android.permission.INTERNET" />
<application android:usesCleartextTraffic="true" ...>
```

`usesCleartextTraffic="true"` 允许HTTP明文传输（开发环境需要，正式上线后改为HTTPS）。

---

## 6. 从模拟到真实AI模型

当前 `server.py` 中使用的是 `mock_audio_detect()` / `mock_video_detect()` 函数，返回随机模拟数据。

要接入真实模型，只需替换这三个函数：

### 6.1 替换音频检测

```python
# 在 server.py 中，找到 mock_audio_detect，替换为:

import torch
import librosa
import numpy as np

# 加载模型（启动时加载一次，不要每次请求都加载）
AASIST_MODEL = None

def load_aasist_model():
    global AASIST_MODEL
    # 从 Exposing-the-Deception 训练产出的 .pth 文件加载
    checkpoint = torch.load("models/aasist_best.pth", map_location="cpu")
    # ... 初始化模型结构，加载权重 ...
    AASIST_MODEL = model
    AASIST_MODEL.eval()

def real_audio_detect(file_bytes: bytes, filename: str) -> dict:
    """真实音频检测"""
    # 1. 保存临时文件
    tmp_path = f"temp/{uuid.uuid4()}.wav"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    # 2. 预处理
    audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
    # 提取LFCC特征...

    # 3. 推理
    with torch.no_grad():
        output = AASIST_MODEL(torch.from_numpy(features).unsqueeze(0))
        score = torch.softmax(output, dim=1)[0][1].item()

    # 4. 清理临时文件
    os.remove(tmp_path)

    return {"score": score, "model_used": "aasist", ...}
```

### 6.2 替换视频检测

```python
# 加载 Exposing-the-Deception 的 EfficientNet 模型
# 或用 DeepfakeDetector 训练出的 best_model-v3.pt

def real_video_detect(file_bytes: bytes, filename: str) -> dict:
    # 1. 保存临时文件
    # 2. 用 cv2 提取视频帧
    # 3. MTCNN/RetinaFace 检测人脸
    # 4. 送入模型推理
    # 5. 汇总所有帧的结果 → 最终评分
    pass
```

### 6.3 替换文本分析

```python
# 用 transformers 加载 BERT 中文模型
# 或调用 GPT-4 API 做深度语义分析

from transformers import BertTokenizer, BertForSequenceClassification

def real_text_analyze(text: str) -> dict:
    # 1. Tokenize
    # 2. BERT推理
    # 3. 得到分类结果 + 置信度
    pass
```

---

## 7. 调试技巧

### 7.1 先用curl测试后端

在接入Android之前，先用curl确认后端能正常工作：
```bash
# 健康检查
curl http://localhost:8000/

# 测试文本分析（最简单，不需要文件）
curl -X POST http://localhost:8000/api/v1/detect/text \
  -H "Content-Type: application/json" \
  -d '{"text":"我是你领导赶紧转50万到这个账号"}'

# 测试音频
curl -X POST http://localhost:8000/api/v1/detect/audio \
  -F "audio=@C:\Users\你的用户名\Music\test.wav"
```

### 7.2 查看后端日志

所有请求都会在终端打印出来：
```
INFO: 127.0.0.1:52341 - "POST /api/v1/detect/text HTTP/1.1" 200 OK
```

### 7.3 Android端调试

在Logcat中过滤 `ApiService` 标签：
```
adb logcat -s ApiService:D
```

### 7.4 手机连不上电脑后端？

常见原因：
1. **防火墙** — Windows防火墙可能拦截8000端口，需要添加入站规则
2. **不在同一WiFi** — 手机必须和电脑连同一个路由器
3. **IP地址变了** — 电脑IP可能变化，检查后更新 `NetworkConfig.java`
4. **Android高版本限制** — Android 9+ 默认禁止HTTP明文传输，我们已在Manifest中加 `usesCleartextTraffic="true"` 解决

**快速排查**：
```bash
# 1. 确认后端在运行
curl http://localhost:8000/

# 2. 用手机浏览器访问
# 手机浏览器打开: http://电脑IP:8000/
# 如果能打开看到JSON，说明网络通
```

---

## 8. 常见问题

### Q1: 后端启动报错 `address already in use`？
```bash
# 端口被占用，换一个端口
python server.py --port 8001
# 对应的 NetworkConfig 也要改成 :8001
```

### Q2: 模拟数据和真实API怎么切换？
不需要改Android代码。后端 `server.py` 里把 `mock_xxx_detect` 换成 `real_xxx_detect` 即可，Android端无感知。

### Q3: 大文件上传很慢怎么办？
- 视频文件建议前端先压缩再上传
- 后端可以改成异步任务：先返回 `task_id`，用户轮询结果
- 当前版本是同步返回（等待推理完成），文件太大会超时

### Q4: 如何部署到公网服务器？
```bash
# 在云服务器上（阿里云/腾讯云）
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:8000

# 前面加 Nginx 反代，配置HTTPS
# Android端 BASE_URL 改为 https://你的域名.com
```

---

## 附录: 项目文件结构

```
多维盾卫-反诈软件开发/
├── backend/
│   ├── server.py           # FastAPI后端主程序
│   └── backend_doc.md      # 本文档
├── deepfake_fight/         # Android项目 (Android Studio打开这个目录)
│   └── app/src/main/java/com/example/deepfake_fight/
│       └── service/
│           ├── ApiService.java       # 网络请求封装
│           └── NetworkConfig.java    # API地址和参数配置
└── 模型复现指南.md          # AI模型训练指南
```

---

> **学妹上手步骤**：
> 1. 先 `pip install fastapi uvicorn python-multipart`
> 2. `python server.py` 启动服务
> 3. 浏览器打开 `http://localhost:8000/docs` 试试各个接口
> 4. 用curl测试通了就OK
> 5. 后续把 `mock_xxx_detect` 函数替换成真实模型
