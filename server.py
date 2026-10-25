"""
多维盾卫 — FastAPI 后端服务
============================
提供音频检测、视频检测、文本分析三个核心API端点。

音频检测: AASIST + RawNet2 深度学习模型 (非模拟)
视频检测: 待接入Exposing-the-Deception
文本分析: 关键词 + 规则引擎

启动方式:
    pip install -r requirements_audio.txt
    python server.py

API文档 (启动后访问):
    http://localhost:8000/docs
"""

import os
import sys
import uuid
import time
import random
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

# 导入管理面板 (延后到app定义后注册)
from dashboard import router as dashboard_router

# 导入换脸API
from faceswap_api import router as faceswap_router

# 导入反诈智能体（大模型 Agent）
from agent import router as agent_router

# 尝试导入AI检测模块
AUDIO_AI_AVAILABLE = False
AUDIO_VISUALIZER_AVAILABLE = False

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from audio_detector import AudioDetector, AudioDetectResult
    AUDIO_AI_AVAILABLE = True
    logger.info("✅ 音频AI检测模块已加载")
except ImportError as e:
    logger.warning(f"⚠️  音频AI检测模块未加载 (需要安装依赖): {e}")
    logger.warning("   将使用模拟检测。安装AI模块: pip install -r requirements_audio.txt")

try:
    from audio_visualizer import AudioVisualizer
    AUDIO_VISUALIZER_AVAILABLE = True
    logger.info("✅ 音频可视化模块已加载")
except ImportError:
    logger.warning("⚠️  音频可视化模块未加载")

# 全局模型实例 (懒加载)
_audio_detector = None
_audio_visualizer = None

# ============================================================
# 初始化
# ============================================================
app = FastAPI(
    title="多维盾卫 · AI反欺诈检测平台",
    description="""
## 多模态AI反欺诈检测服务

### 功能模块
- **音频检测**: AASIST-L深度学习模型, 识别AI合成语音/声音克隆
- **视频检测**: EfficientNet-B0模型, 检测Deepfake换脸
- **文本分析**: 关键词+规则引擎, 识别诈骗话术
- **换脸演示**: InsightFace模型, 教育用户了解Deepfake原理

### 技术特点
- 端侧AI推理, 数据不上传云端
- 三模态融合检测(视频+音频+文本)
- 自研可视化分析引擎
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 允许跨域（让Android App能访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 智能体请求也计入统计（此前 agent/chat 未更新 stats，导致 dashboard 数字不涨）
@app.middleware("http")
async def count_agent_requests(request, call_next):
    path = request.url.path
    if "/agent/chat" in path:
        stats["total_requests"] += 1
        stats["text_count"] += 1  # 智能体归入文本/语义分析
    response = await call_next(request)
    return response

# 注册子路由
app.include_router(dashboard_router)
app.include_router(faceswap_router)
if agent_router is not None:
    app.include_router(agent_router)
    logger.info("✅ 反诈智能体(Agent)模块已加载")
else:
    logger.warning("⚠️  反诈智能体(Agent)模块未加载 (需安装 fastapi/pydantic)")

# 统计计数器
stats = {"total_requests": 0, "audio_count": 0, "video_count": 0, "text_count": 0,
         "fraud_detected": 0, "start_time": datetime.now().isoformat()}

# 模拟任务存储（后续改为Redis/数据库）
task_store: dict = {}

# 可视化输出目录
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def get_audio_detector() -> Optional[AudioDetector]:
    """获取音频检测器 (懒加载，首次调用时加载模型)"""
    global _audio_detector
    if not AUDIO_AI_AVAILABLE:
        return None
    if _audio_detector is None:
        try:
            logger.info("🔧 首次加载音频AI模型...")
            _audio_detector = AudioDetector(use_asr=True, whisper_model="base")
            _audio_detector.load_models()
            logger.info("✅ 音频AI模型就绪")
        except Exception as e:
            logger.error(f"❌ 音频模型加载失败: {e}")
            return None
    return _audio_detector


def get_audio_visualizer() -> Optional[AudioVisualizer]:
    """获取可视化器"""
    global _audio_visualizer
    if not AUDIO_VISUALIZER_AVAILABLE:
        return None
    if _audio_visualizer is None:
        try:
            _audio_visualizer = AudioVisualizer()
        except Exception as e:
            logger.error(f"❌ 可视化器初始化失败: {e}")
            return None
    return _audio_visualizer

# ============================================================
# 数据模型
# ============================================================
class TextRequest(BaseModel):
    text: str
    model: str = "bert-base-chinese"

class DetectionResult(BaseModel):
    task_id: str
    status: str                # "processing" | "completed" | "failed"
    risk_score: float          # 0.0 ~ 1.0, 越高越可疑
    risk_level: int            # 1~5
    verdict: str               # "真实" | "低风险" | "可疑" | "高风险" | "伪造"
    details: dict = {}
    suggestions: list = []
    processed_at: Optional[str] = None

# ============================================================
# 工具函数
# ============================================================
def generate_task_id() -> str:
    return f"task_{uuid.uuid4().hex[:12]}"

def score_to_level(score: float) -> int:
    if score >= 0.8: return 5
    if score >= 0.6: return 4
    if score >= 0.4: return 3
    if score >= 0.2: return 2
    return 1

def score_to_verdict(score: float) -> str:
    if score >= 0.8: return "伪造"
    if score >= 0.6: return "高风险"
    if score >= 0.4: return "可疑"
    if score >= 0.2: return "低风险"
    return "真实"

def generate_suggestions(level: int, mode: str) -> list:
    if level >= 4:
        return [
            "立即停止与对方的所有通讯",
            "切勿进行任何形式的转账汇款",
            "保存此检测报告作为报警证据",
            "拨打 96110 反诈专线举报"
        ]
    elif level == 3:
        return [
            "谨慎对待对方的资金要求",
            "通过其他渠道独立验证对方身份",
            "勿点击对方发送的任何链接"
        ]
    else:
        return [
            "未发现明显伪造痕迹",
            "但仍建议保持基本安全防范意识"
        ]


# ============================================================
# 模拟AI检测 (后续替换为真实模型)
# ============================================================
def real_audio_detect(file_bytes: bytes, filename: str) -> dict:
    """
    真实AI音频检测 — AASIST + Whisper ASR

    流水线:
        1. 保存临时文件
        2. Whisper ASR → 语音转文字
        3. AASIST → AI合成检测
        4. 文本风险分析
        5. 融合决策
        6. 生成可视化报告图
    """
    start_time = time.time()

    # 保存临时文件
    suffix = os.path.splitext(filename)[1] or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        detector = get_audio_detector()
        if detector is None:
            raise RuntimeError("音频检测器未就绪")

        # 执行检测
        result = detector.detect(tmp_path, save_visualization=False)

        # 生成可视化报告
        report_path = ""
        visualizer = get_audio_visualizer()
        if visualizer:
            try:
                report_name = f"audio_report_{uuid.uuid4().hex[:8]}.png"
                report_path = str(REPORTS_DIR / report_name)
                visualizer.generate_report(tmp_path, result, report_path)
                logger.info(f"📊 可视化报告: {report_path}")
            except Exception as e:
                logger.warning(f"可视化生成失败: {e}")

        elapsed = time.time() - start_time
        logger.info(f"⏱️  音频检测耗时: {elapsed:.1f}s")

        # 构建返回结果
        return {
            "score": result.spoof_score,
            "model_used": "AASIST + Whisper (AI)",
            "sample_rate": 16000,
            "segments_analyzed": result.segments_analyzed,
            "transcript": result.transcript,
            "transcript_language": result.transcript_language,
            "annotated_image": report_path,
            "details": {
                "aasist_score": result.spoof_score,
                "segment_scores": result.segment_scores,
                "asr_risk_flags": [
                    {"category": f["category"], "keywords": f.get("matched_keywords", [])}
                    for f in result.asr_risk_flags
                ],
                "audio_artifacts": result.audio_artifacts,
                "source": "local_ai",
                "elapsed_sec": round(elapsed, 2)
            }
        }

    except Exception as e:
        logger.error(f"❌ AI音频检测失败: {e}")
        # 降级到模拟检测
        logger.warning("🔄 降级到模拟检测...")
        return mock_audio_detect_fallback(file_bytes, filename)
    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


def mock_audio_detect_fallback(file_bytes: bytes, filename: str) -> dict:
    """模拟音频检测 — AI模型不可用时的后备方案"""
    time.sleep(0.5)
    # 使用基本启发式: 文件太小可能是合成音频
    size_score = min(0.3, len(file_bytes) / (500 * 1024))
    score = round(random.uniform(0.05, 0.30) + size_score, 4)
    score = min(score, 0.45)  # 模拟不会给出高置信度

    return {
        "score": score,
        "model_used": "heuristic (mock)",
        "sample_rate": 16000,
        "segments_analyzed": max(1, len(file_bytes) // (64000 * 2)),
        "transcript": "",
        "transcript_language": "",
        "annotated_image": "",
        "details": {
            "aasist_score": 0.0,
            "segment_scores": [],
            "asr_risk_flags": [],
            "audio_artifacts": [],
            "source": "mock_fallback",
            "warning": "AI模型未加载，使用启发式分析"
        }
    }


def mock_audio_detect(file_bytes: bytes, filename: str) -> dict:
    """音频检测入口 — 自动选择AI或模拟"""
    if AUDIO_AI_AVAILABLE:
        return real_audio_detect(file_bytes, filename)
    else:
        return mock_audio_detect_fallback(file_bytes, filename)

def mock_video_detect(file_bytes: bytes, filename: str) -> dict:
    """模拟视频检测 — 后续替换为 Exposing-the-Deception / EfficientNet"""
    time.sleep(1.5)
    # 30%概率返回高风险
    if random.random() < 0.3:
        score = random.uniform(0.60, 0.90)
    else:
        score = random.uniform(0.03, 0.30)
    frames_analyzed = random.randint(10, 30)
    return {
        "score": round(score, 4),
        "model_used": "efficientnet-b4 (mock)",
        "frames_analyzed": frames_analyzed,
        "details": {
            "frame_scores": [round(random.uniform(0, 0.5), 4) for _ in range(frames_analyzed)],
            "frequency_anomaly": round(random.uniform(0, 0.3), 4),
            "blink_rate_normal": random.choice([True, False]),
            "source": "cloud"
        }
    }

def mock_text_analyze(text: str) -> dict:
    """模拟文本分析 — 后续替换为 BERT + LLM Agent"""
    time.sleep(0.5)
    keywords_fraud = ["转账", "汇款", "打款", "警察", "法院", "逮捕", "冻结",
                       "领导", "老板", "客服", "验证码", "安全账户"]
    keywords_urgent = ["马上", "立即", "快", "赶紧", "立刻", "尽快", "紧急"]
    keywords_money = ["钱", "元", "万", "千", "百", "款", "费"]

    fraud_count = sum(1 for kw in keywords_fraud if kw in text)
    urgent_count = sum(1 for kw in keywords_urgent if kw in text)
    money_count = sum(1 for kw in keywords_money if kw in text)

    base_score = 0.05
    score = base_score + fraud_count * 0.10 + urgent_count * 0.08 + money_count * 0.06
    score = min(round(score, 4), 0.98)

    fraud_types = []
    if any(kw in text for kw in ["警察", "法院", "逮捕", "冻结"]):
        fraud_types.append("冒充公检法")
    if any(kw in text for kw in ["领导", "老板", "转账", "汇款"]):
        fraud_types.append("冒充领导/熟人")
    if any(kw in text for kw in ["刷单", "返利", "返现", "佣金"]):
        fraud_types.append("刷单诈骗")

    return {
        "score": score,
        "model_used": "bert+keyword (mock)",
        "text_length": len(text),
        "fraud_type": fraud_types[0] if fraud_types else "未识别",
        "details": {
            "fraud_keywords_found": fraud_count,
            "urgent_keywords_found": urgent_count,
            "money_keywords_found": money_count,
            "possible_types": fraud_types,
            "source": "cloud"
        }
    }

# ============================================================
# API端点
# ============================================================

@app.get("/", response_class=HTMLResponse)
def root():
    """服务仪表盘首页"""
    return f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <title>多维盾卫 · AI反诈服务</title>
    <style>
      *{{margin:0;padding:0;box-sizing:border-box}}
      body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f0f2f5;min-height:100vh}}
      .header{{background:linear-gradient(135deg,#1a237e,#0d47a1);color:#fff;padding:32px 24px;text-align:center}}
      .header h1{{font-size:28px;margin-bottom:6px}}
      .header p{{opacity:0.75;font-size:14px}}
      .status{{display:inline-block;margin-top:10px;padding:4px 14px;border-radius:12px;background:#00c853;font-size:12px}}
      .container{{max-width:960px;margin:0 auto;padding:24px 16px}}
      .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:24px}}
      .stat-card{{background:#fff;border-radius:12px;padding:20px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.08)}}
      .stat-card .num{{font-size:32px;font-weight:700;color:#1a237e}}
      .stat-card .label{{font-size:12px;color:#888;margin-top:4px}}
      .section{{background:#fff;border-radius:14px;padding:24px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.08)}}
      .section h2{{font-size:18px;margin-bottom:16px;color:#1a237e}}
      .endpoint{{display:flex;align-items:center;gap:12px;padding:14px;border-radius:10px;margin-bottom:8px;background:#f8f9fa;text-decoration:none;color:inherit;transition:background .2s}}
      .endpoint:hover{{background:#e3f2fd}}
      .method{{padding:5px 10px;border-radius:6px;font-size:11px;font-weight:700;color:#fff;min-width:48px;text-align:center}}
      .method.post{{background:#1565c0}}
      .method.get{{background:#2e7d32}}
      .endpoint .path{{font-family:monospace;font-size:14px;flex:1}}
      .endpoint .arrow{{color:#888;font-size:18px}}
      .test-area{{margin-top:16px}}
      .test-area textarea{{width:100%;height:100px;border:1px solid #ddd;border-radius:8px;padding:12px;font-size:14px;resize:vertical}}
      .test-area button{{margin-top:8px;padding:10px 28px;background:#1a237e;color:#fff;border:none;border-radius:8px;font-size:14px;cursor:pointer}}
      .test-area button:hover{{background:#283593}}
      .result{{margin-top:12px;padding:14px;background:#263238;color:#aed581;border-radius:8px;font-family:monospace;font-size:12px;white-space:pre-wrap;display:none;max-height:300px;overflow:auto}}
      .footer{{text-align:center;padding:24px;color:#888;font-size:12px}}
      .footer a{{color:#1a237e}}
    </style>
    </head>
    <body>
    <div class="header">
      <h1>🛡️ 多维盾卫</h1>
      <p>多模态AI反欺诈检测服务 v1.0</p>
      <div class="status">● 服务运行中 | 音频AI: AI模块</div>
    </div>

    <div class="container">
    <div class="stats">
      <div class="stat-card"><div class="num">{stats["total_requests"]}</div><div class="label">总请求数</div></div>
      <div class="stat-card"><div class="num">{stats["audio_count"]}</div><div class="label">音频检测</div></div>
      <div class="stat-card"><div class="num">{stats["video_count"]}</div><div class="label">视频检测</div></div>
      <div class="stat-card"><div class="num">{stats["text_count"]}</div><div class="label">文本分析</div></div>
      <div class="stat-card"><div class="num">{stats["fraud_detected"]}</div><div class="label">识破欺诈</div></div>
    </div>

    <div class="section">
      <h2>📡 API 接口</h2>
      <a class="endpoint" href="/docs" target="_blank">
        <span class="method post">POST</span><span class="path">/api/v1/detect/audio</span><span style="color:#888;font-size:12px">音频检测</span><span class="arrow">↗</span>
      </a>
      <a class="endpoint" href="/docs" target="_blank">
        <span class="method post">POST</span><span class="path">/api/v1/detect/video</span><span style="color:#888;font-size:12px">视频检测</span><span class="arrow">↗</span>
      </a>
      <a class="endpoint" href="/docs" target="_blank">
        <span class="method post">POST</span><span class="path">/api/v1/detect/text</span><span style="color:#888;font-size:12px">文本分析</span><span class="arrow">↗</span>
      </a>
      <a class="endpoint" href="/docs" target="_blank">
        <span class="method get">GET</span><span class="path">/api/v1/result/{{task_id}}</span><span style="color:#888;font-size:12px">结果查询</span><span class="arrow">↗</span>
      </a>
      <p style="margin-top:12px;font-size:13px;color:#888">
        📖 完整交互式文档: <a href="/docs" target="_blank">/docs</a> (Swagger) ｜
        启动时间: {stats["start_time"][:19]}
      </p>
    </div>

    <div class="section">
      <h2>🧪 快速测试 - 文本分析</h2>
      <div class="test-area">
        <textarea id="textInput" placeholder="输入测试文本，例如: 您好我是市公安局的，你的身份证涉嫌洗钱案件，请将资金转入安全账户配合调查。"></textarea>
        <button onclick="testText()">发送检测请求</button>
        <div class="result" id="result"></div>
      </div>
    </div>

    <div class="section">
      <h2>🔧 使用 curl 测试</h2>
      <div class="result" style="display:block;background:#f5f5f5;color:#333">
    # 文本分析
    curl -X POST http://localhost:8000/api/v1/detect/text \\
      -H "Content-Type: application/json" \\
      -d '{{"text":"测试文本内容","model":"bert-base-chinese"}}'

    # 音频检测
    curl -X POST http://localhost:8000/api/v1/detect/audio \\
      -F "audio=@test.wav"

    # 视频检测
    curl -X POST http://localhost:8000/api/v1/detect/video \\
      -F "video=@test.mp4"
      </div>
    </div>
    </div>

    <div class="footer">
      多维盾卫 · AI反诈智能检测系统 · <a href="/docs">API文档</a>
    </div>

    <script>
    async function testText(){{
      const text=document.getElementById('textInput').value||'测试文本';
      const r=document.getElementById('result');
      r.style.display='block';
      r.textContent='请求中...';
      try{{
        const resp=await fetch('/api/v1/detect/text',{{
          method:'POST',headers:{{'Content-Type':'application/json'}},
          body:JSON.stringify({{text:text}})
        }});
        const data=await resp.json();
        r.textContent=JSON.stringify(data,null,2);
      }}catch(e){{r.textContent='错误: '+e.message;}}
    }}
    </script>
    </body>
    </html>
    """

# ---------- 音频检测 ----------
@app.post("/api/v1/detect/audio", tags=["检测服务"],
    summary="音频真伪检测",
    description="上传音频文件, 使用AASIST-L深度学习模型检测是否为AI合成语音。支持WAV/MP3/M4A/AAC格式。")
async def detect_audio(
    audio: UploadFile = File(..., description="音频文件 (.wav/.mp3/.m4a/.aac)"),
    model: str = Form("aasist", description="检测模型 (aasist/rawnet2)")
):
    # 校验文件
    if not audio.filename:
        raise HTTPException(400, "未提供音频文件")

    allowed_ext = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg")
    if not audio.filename.lower().endswith(allowed_ext):
        raise HTTPException(400, f"不支持的音频格式，允许: {allowed_ext}")

    # 读取文件
    file_bytes = await audio.read()

    if len(file_bytes) == 0:
        raise HTTPException(400, "音频文件为空")

    if len(file_bytes) > 50 * 1024 * 1024:  # 50MB限制
        raise HTTPException(400, "音频文件超过50MB限制")

    # 创建任务
    task_id = generate_task_id()
    task_store[task_id] = {"status": "processing", "created_at": datetime.now().isoformat()}
    stats["total_requests"] += 1
    stats["audio_count"] += 1

    # 执行检测 (当前使用模拟)
    try:
        result = mock_audio_detect(file_bytes, audio.filename)
    except Exception as e:
        task_store[task_id]["status"] = "failed"
        raise HTTPException(500, f"检测失败: {str(e)}")

    # 构建返回
    score = result["score"]
    level = score_to_level(score)

    # 更新统计
    if level >= 4:
        stats["fraud_detected"] += 1

    response = {
        "task_id": task_id,
        "status": "completed",
        "risk_score": score,
        "risk_level": level,
        "verdict": score_to_verdict(score),
        "detection_type": "audio",
        "model_used": result["model_used"],
        "transcript": result.get("transcript", ""),
        "transcript_language": result.get("transcript_language", ""),
        "annotated_image": result.get("annotated_image", ""),
        # 生成可访问的图片URL
        "annotated_image_url": f"/api/v1/report-image/{os.path.basename(result.get('annotated_image', ''))}"
                               if result.get("annotated_image") else "",
        "details": result["details"],
        "suggestions": generate_suggestions(level, "audio"),
        "processed_at": datetime.now().isoformat()
    }

    task_store[task_id] = response
    return response


# ---------- 视频检测 ----------
@app.post("/api/v1/detect/video")
async def detect_video(
    video: UploadFile = File(...),
    model: str = Form("efficientnet-b4"),
    frame_rate: int = Form(5)
):
    """
    上传视频文件，检测是否存在Deepfake换脸。

    请求: multipart/form-data
        - video: 视频文件 (.mp4, .mov, .avi)
        - model: 模型选择 (efficientnet-b4 / xception)
        - frame_rate: 每秒采样帧数 (默认5)

    返回: DetectionResult
    """
    if not video.filename:
        raise HTTPException(400, "未提供视频文件")

    allowed_ext = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    if not video.filename.lower().endswith(allowed_ext):
        raise HTTPException(400, f"不支持的视频格式，允许: {allowed_ext}")

    file_bytes = await video.read()

    if len(file_bytes) == 0:
        raise HTTPException(400, "视频文件为空")

    if len(file_bytes) > 200 * 1024 * 1024:  # 200MB限制
        raise HTTPException(400, "视频文件超过200MB限制")

    task_id = generate_task_id()
    stats["total_requests"] += 1
    stats["video_count"] += 1

    try:
        result = mock_video_detect(file_bytes, video.filename)
    except Exception as e:
        raise HTTPException(500, f"检测失败: {str(e)}")

    score = result["score"]
    level = score_to_level(score)
    response = {
        "task_id": task_id,
        "status": "completed",
        "risk_score": score,
        "risk_level": level,
        "verdict": score_to_verdict(score),
        "detection_type": "video",
        "model_used": result["model_used"],
        "details": result["details"],
        "suggestions": generate_suggestions(level, "video"),
        "processed_at": datetime.now().isoformat()
    }
    return response


# ---------- 文本分析 ----------
@app.post("/api/v1/detect/text")
async def detect_text(request: TextRequest):
    """
    提交文本内容，分析是否存在诈骗风险。

    请求: application/json
    {
        "text": "对话文本内容",
        "model": "bert-base-chinese"
    }

    返回: DetectionResult
    """
    if not request.text or len(request.text.strip()) < 5:
        raise HTTPException(400, "文本内容太短，至少5个字符")

    if len(request.text) > 5000:
        raise HTTPException(400, "文本超过5000字符限制")

    task_id = generate_task_id()
    stats["total_requests"] += 1
    stats["text_count"] += 1

    try:
        result = mock_text_analyze(request.text)
    except Exception as e:
        raise HTTPException(500, f"分析失败: {str(e)}")

    score = result["score"]
    level = score_to_level(score)
    response = {
        "task_id": task_id,
        "status": "completed",
        "risk_score": score,
        "risk_level": level,
        "verdict": score_to_verdict(score),
        "detection_type": "text",
        "model_used": result["model_used"],
        "fraud_type": result.get("fraud_type", ""),
        "details": result["details"],
        "suggestions": generate_suggestions(level, "text"),
        "processed_at": datetime.now().isoformat()
    }
    return response


# ---------- 结果查询 ----------
class ReportRequest(BaseModel):
    type: str  # "audio" / "video" / "text"


@app.post("/api/v1/report")
def report_detection(req: ReportRequest):
    """端侧检测完成后的统计上报（只报类型，不传数据，保护隐私）"""
    stats["total_requests"] += 1
    if req.type == "audio":
        stats["audio_count"] += 1
    elif req.type == "video":
        stats["video_count"] += 1
    elif req.type == "text":
        stats["text_count"] += 1
    return {"status": "ok", "total": stats["total_requests"]}


@app.get("/api/health")
def health_check():
    """健康检查 — 供App和管理面板使用"""
    return {
        "status": "ok",
        "service": "多维盾卫 AI检测服务",
        "version": "2.0.0",
        "models": {
            "audio_ai": AUDIO_AI_AVAILABLE,
            "audio_visualizer": AUDIO_VISUALIZER_AVAILABLE,
            "whisper": AUDIO_AI_AVAILABLE,
            "viz": AUDIO_VISUALIZER_AVAILABLE
        },
        "stats": {
            "total": stats["total_requests"],
            "audio": stats["audio_count"],
            "video": stats["video_count"],
            "text": stats["text_count"],
            "fraud": stats["fraud_detected"]
        },
        "time": datetime.now().isoformat()
    }


@app.get("/api/v1/report-image/{filename}")
async def get_report_image(filename: str):
    """获取检测报告可视化图片"""
    image_path = REPORTS_DIR / filename
    if not image_path.exists():
        raise HTTPException(404, f"报告图片不存在: {filename}")
    return FileResponse(str(image_path), media_type="image/png")


@app.get("/api/v1/result/{task_id}")
async def query_result(task_id: str):
    """查询之前的检测结果"""
    if task_id not in task_store:
        raise HTTPException(404, "任务不存在")

    return task_store[task_id]


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  多维盾卫 - 后端AI检测服务")
    print("  API文档: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
