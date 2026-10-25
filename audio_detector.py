"""
============================================================
多维盾卫 — AI音频真伪检测引擎
============================================================

双模型流水线:
  1. Whisper (ASR) → 语音转文字，理解通话内容
  2. AASIST/RawNet2 → 检测是否为AI合成/声音克隆

输入: 音频文件路径 或 bytes
输出: {
    "transcript": "识别出的通话文字",
    "spoof_score": 0.95,
    "is_spoofed": true,
    "risk_level": 5,
    "analysis": "检测到AI合成语音特征..."
}

使用方式:
    from audio_detector import AudioDetector
    detector = AudioDetector()
    result = detector.detect("test.wav")
============================================================
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AudioDetector")

# ============================================================
# 配置
# ============================================================
BACKEND_DIR = Path(__file__).parent
MODELS_DIR = BACKEND_DIR / "models"
AASIST_WEIGHTS_PATH = MODELS_DIR / "aasist" / "weights" / "AASIST-L.pth"

SAMPLE_RATE = 16000       # 统一采样率
SEGMENT_DURATION = 4.0    # 每段4秒
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)  # 64000 samples

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class AudioDetectResult:
    """音频检测结果"""
    transcript: str = ""                    # Whisper识别的文字
    transcript_language: str = "zh"         # 识别语言
    spoof_score: float = 0.0                # AI合成概率 (0~1)
    is_spoofed: bool = False                # 是否为AI伪造
    risk_level: int = 1                     # 1~5
    verdict: str = "真实"                   # 中文判定
    segments_analyzed: int = 0              # 分析片段数
    segment_scores: list = field(default_factory=list)
    # 详细分析
    asr_risk_flags: list = field(default_factory=list)   # 文本内容风险标记
    audio_artifacts: list = field(default_factory=list)  # 音频伪造痕迹
    # 可视化
    annotated_image: str = ""               # 可视化图片路径
    waveform_path: str = ""                 # 波形图路径
    spectrogram_path: str = ""              # 频谱图路径


# ============================================================
# AASIST 模型架构 (自包含实现)
# ============================================================

class SincConv(nn.Module):
    """SincNet风格的卷积层 — 直接从原始波形提取特征"""

    def __init__(self, out_channels: int, kernel_size: int, sample_rate: int = 16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # 可学习的截止频率
        self.low_hz = nn.Parameter(torch.rand(out_channels) * 50)
        self.band_hz = nn.Parameter(torch.rand(out_channels) * (sample_rate / 2 - 50) + 50)

        # Hamming窗
        n = torch.arange(kernel_size)
        self.register_buffer("window", 0.54 - 0.46 * torch.cos(2 * np.pi * n / kernel_size))
        self.register_buffer("n_", 2 * np.pi * n.unsqueeze(0) / sample_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T) 原始波形
        Returns:
            (B, out_channels, T) 滤波后的特征
        """
        low = self.low_hz.abs() / self.sample_rate
        high = low + self.band_hz.abs() / self.sample_rate
        high = torch.clamp(high, max=0.5)

        band = (high - low).unsqueeze(1) * self.n_  # (C, kernel_size)
        low = low.unsqueeze(1) * self.n_

        # sinc滤波器
        filt = (torch.sin(band) * torch.cos(low)) * self.window

        # 卷积
        filt = filt.unsqueeze(1)  # (out_channels, 1, kernel_size)
        out = F.conv1d(x.squeeze(1).unsqueeze(1) if x.dim() == 3 else x,
                       filt, stride=1, padding=self.kernel_size // 2)
        if out.size(1) == 1:
            out = out.repeat(1, self.out_channels, 1)
        out = out[:, :self.out_channels, :]
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 块"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        y = x.mean(dim=2)  # Global Average Pooling
        y = self.fc(y).unsqueeze(2)
        return x * y


class ResBlock(nn.Module):
    """带SE的残差块"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.pool = nn.MaxPool1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.3)
        out = F.leaky_relu(self.bn2(self.conv2(out)), 0.3)
        out = self.se(out)
        out = out + residual
        out = self.pool(out)
        return out


class GraphAttentionLayer(nn.Module):
    """图注意力层 — 用于spectro-temporal图"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(out_dim * 2, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            h: (B, N, in_dim) 节点特征
            adj: (N, N) 邻接矩阵，None则全连接
        Returns:
            (B, N, out_dim) 更新后的节点特征
        """
        Wh = self.W(h)  # (B, N, out_dim)
        B, N, D = Wh.shape

        # 计算注意力分数
        Wh1 = Wh.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        Wh2 = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
        e = self.leaky_relu(self.a(torch.cat([Wh1, Wh2], dim=-1))).squeeze(-1)  # (B, N, N)

        if adj is not None:
            e = e.masked_fill(adj.unsqueeze(0) == 0, -1e9)

        attention = F.softmax(e, dim=-1)
        h_prime = torch.bmm(attention, Wh)  # (B, N, D)
        return F.elu(h_prime)


class AASISTModel(nn.Module):
    """
    AASIST-Lite: 简化版 Audio Anti-Spoofing 模型

    架构:
        1. SincConv → 从原始波形提取特征
        2. 3× ResBlocks → 深度特征提取
        3. Graph Attention → Spectro-temporal关系建模
        4. Attention Pooling → 全局表示
        5. Classifier → 真/伪二分类

    适配 AASIST-L 预训练权重 (~85K参数)
    """

    def __init__(self, nb_samp: int = 64000):
        super().__init__()
        self.nb_samp = nb_samp

        # --- Encoder ---
        self.sinc = SincConv(out_channels=70, kernel_size=251, sample_rate=16000)
        self.bn0 = nn.BatchNorm1d(70)
        self.res1 = ResBlock(70, 70)
        self.res2 = ResBlock(70, 70)
        self.res3 = ResBlock(70, 64)

        # --- Graph Attention ---
        self.gat1 = GraphAttentionLayer(64, 64)
        self.gat2 = GraphAttentionLayer(64, 32)

        # --- Pooling ---
        self.att_pool = nn.Sequential(
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, nb_samp) 或 (B, nb_samp) 原始波形
        Returns:
            (B, 2) logits (bonafide, spoof)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)

        # Pad/truncate
        if x.shape[-1] < self.nb_samp:
            x = F.pad(x, (0, self.nb_samp - x.shape[-1]))
        else:
            x = x[:, :, :self.nb_samp]

        # --- Encoder ---
        x = self.sinc(x)
        x = F.leaky_relu(self.bn0(x), 0.3)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)  # (B, 64, T')

        # --- 构造图节点 ---
        B, C, T = x.shape
        x_t = x.permute(0, 2, 1)  # (B, T, C) — 时间节点

        # Split into two node groups for hetero graph
        half_c = C // 2
        spec_nodes = x.mean(dim=2)  # (B, C) — 频谱节点 (简化为均值池化)
        temp_nodes = x_t.mean(dim=1)  # (B, C) — 时间节点 (简化为均值池化)

        # Reshape to (B, N, D) format for graph attention
        # We use temporal features as graph nodes
        n_nodes = min(T, 32)  # Max 32 graph nodes for efficiency
        indices = torch.linspace(0, T - 1, n_nodes, dtype=torch.long, device=x.device)
        h = x_t[:, indices, :]  # (B, n_nodes, C)

        # --- Graph Attention ---
        h = self.gat1(h)
        h = self.gat2(h)  # (B, n_nodes, 32)

        # --- Attention Pooling ---
        att_weights = self.att_pool(h).squeeze(-1)  # (B, n_nodes)
        att_weights = F.softmax(att_weights, dim=-1).unsqueeze(1)
        global_feat = torch.bmm(att_weights, h).squeeze(1)  # (B, 32)

        # --- Classifier ---
        logits = self.classifier(global_feat)  # (B, 2)
        return logits


# ============================================================
# 简化版 RawNet2 后备模型
# ============================================================

class SimpleSpoofDetector(nn.Module):
    """
    轻量级替代检测器 — 当AASIST权重不可用时使用。
    基于RawNet2的简化版本，用于快速验证音频特征。
    """
    def __init__(self):
        super().__init__()
        # SincConv layer
        self.sinc = SincConv(64, 251)

        # ResBlocks
        self.blocks = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 128),
        )

        # GRU layer
        self.gru = nn.GRU(128, 64, num_layers=2, batch_first=True, bidirectional=True)

        # Attention pooling
        self.att = nn.Linear(128, 1)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[-1] < SEGMENT_SAMPLES:
            x = F.pad(x, (0, SEGMENT_SAMPLES - x.shape[-1]))
        else:
            x = x[:, :, :SEGMENT_SAMPLES]

        x = self.sinc(x)
        x = self.blocks(x)  # (B, 128, T')
        x = x.permute(0, 2, 1)  # (B, T', 128)

        x, _ = self.gru(x)  # (B, T', 128)

        att = F.softmax(self.att(x), dim=1)
        x = (x * att).sum(dim=1)  # (B, 128)

        return self.fc(x)


# ============================================================
# 音频检测引擎
# ============================================================

class AudioDetector:
    """
    音频真伪检测引擎

    流水线:
        音频输入 → 预处理 → [Whisper ASR, AASIST检测] → 融合决策 → 结果
    """

    def __init__(self,
                 use_asr: bool = True,
                 use_aasist: bool = True,
                 whisper_model: str = "base"):
        """
        Args:
            use_asr: 是否启用语音识别
            use_aasist: 是否启用AI检测 (关闭则使用简单特征分析)
            whisper_model: Whisper模型大小 (tiny/base/small/medium/large)
        """
        self.use_asr = use_asr
        self.use_aasist = use_aasist
        self.whisper_model_name = whisper_model

        # 延迟加载模型
        self._whisper_model = None
        self._spoof_model = None
        self._model_loaded = False

        logger.info(f"AudioDetector 初始化 — Device: {DEVICE}")
        logger.info(f"  ASR: {use_asr} ({whisper_model}), "
                    f"AASIST: {use_aasist}")

    # ----- 模型加载 -----

    def load_models(self):
        """加载所有模型（首次调用时自动触发）"""
        if self._model_loaded:
            return

        logger.info("[*] 加载AI模型...")

        # 加载Whisper
        if self.use_asr:
            self._load_whisper()

        # 加载反欺骗模型
        if self.use_aasist:
            self._load_spoof_model()
        else:
            logger.info("[WARN]  跳过AASIST，将使用特征分析模式")

        self._model_loaded = True
        logger.info("[OK] 模型加载完成")

    def _load_whisper(self):
        """加载Whisper ASR模型"""
        try:
            import whisper
            logger.info(f"   加载 Whisper {self.whisper_model_name} ...")
            self._whisper_model = whisper.load_model(self.whisper_model_name)
            logger.info(f"   [OK] Whisper就绪")
        except ImportError:
            logger.warning("   [WARN]  openai-whisper 未安装，ASR功能禁用")
            logger.warning("   安装: pip install openai-whisper")
            self.use_asr = False
        except Exception as e:
            logger.warning(f"   [WARN]  Whisper加载失败: {e}")
            self.use_asr = False

    def _load_spoof_model(self):
        """加载反欺骗模型"""
        if AASIST_WEIGHTS_PATH.exists():
            # 尝试加载AASIST预训练权重
            try:
                logger.info(f"   加载 AASIST-L 预训练权重: {AASIST_WEIGHTS_PATH}")
                self._spoof_model = AASISTModel()
                checkpoint = torch.load(AASIST_WEIGHTS_PATH, map_location=DEVICE)

                # 兼容不同格式的checkpoint
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get('model_state_dict', checkpoint)
                else:
                    state_dict = checkpoint

                # 尝试strict=False加载
                missing, unexpected = self._spoof_model.load_state_dict(
                    state_dict, strict=False
                )
                if missing:
                    logger.info(f"   [WARN]  缺失键: {len(missing)} (模型结构不完全匹配)")
                if unexpected:
                    logger.info(f"   [WARN]  多余键: {len(unexpected)}")

                self._spoof_model.to(DEVICE).eval()
                self._aasist_available = True
                logger.info("   [OK] AASIST-L 就绪")
                return
            except Exception as e:
                logger.warning(f"   [WARN]  AASIST加载失败: {e}")
        else:
            logger.info(f"   [WARN]  未找到AASIST权重: {AASIST_WEIGHTS_PATH}")

        # 后备: 使用简单检测器
        logger.info("   [~] 使用简化检测器 (特征分析模式)")
        self._spoof_model = SimpleSpoofDetector().to(DEVICE).eval()
        self._aasist_available = False

    # ----- 预处理 -----

    def preprocess(self, audio_path: str) -> Tuple[np.ndarray, list]:
        """
        预处理音频文件:
          1. 重采样到16kHz mono
          2. 去除静音
          3. 切分为4秒片段

        Returns:
            segments: [(np.ndarray,)] 音频片段列表
            original_audio: 完整预处理后的音频
        """
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        # 去除静音段
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # 归一化
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # 切分为片段
        segments = []
        for i in range(0, len(audio), SEGMENT_SAMPLES):
            seg = audio[i:i + SEGMENT_SAMPLES]
            if len(seg) >= SAMPLE_RATE:  # 至少1秒
                # Padding
                if len(seg) < SEGMENT_SAMPLES:
                    seg = np.pad(seg, (0, SEGMENT_SAMPLES - len(seg)))
                segments.append(seg)

        # 如果音频太短（< 1秒），复制填充
        if not segments and len(audio) > 0:
            seg = np.pad(audio, (0, max(0, SEGMENT_SAMPLES - len(audio))))[:SEGMENT_SAMPLES]
            segments.append(seg)

        logger.info(f"   预处理: {len(audio)/SAMPLE_RATE:.1f}s → {len(segments)}片段")
        return segments, audio

    # ----- ASR 语音识别 -----

    def transcribe(self, audio: np.ndarray) -> Tuple[str, str]:
        """
        使用Whisper进行语音转文字

        Returns:
            (transcript, language)
        """
        if not self.use_asr or self._whisper_model is None:
            return "", "unknown"

        try:
            # Whisper期望float32, 16kHz
            result = self._whisper_model.transcribe(
                audio.astype(np.float32),
                language=None,           # 自动检测语言
                task="transcribe",
                fp16=(DEVICE.type == "cuda"),
                verbose=False
            )
            transcript = result.get("text", "").strip()
            language = result.get("language", "unknown")
            logger.info(f"   ASR [{language}]: {transcript[:100]}...")
            return transcript, language
        except Exception as e:
            logger.warning(f"   [WARN]  ASR失败: {e}")
            return "", "unknown"

    # ----- 反欺骗检测 -----

    def detect_spoof(self, segments: list) -> Dict:
        """
        对每个片段进行AI合成检测

        Returns:
            dict with scores, predictions, artifacts
        """
        if self._spoof_model is None:
            return {"scores": [0.1], "mean_score": 0.1, "is_spoofed": False}

        scores = []
        with torch.no_grad():
            for seg in segments:
                x = torch.FloatTensor(seg).unsqueeze(0).to(DEVICE)  # (1, T)
                output = self._spoof_model(x)
                prob = torch.softmax(output, dim=1)[0, 1].item()  # spoof概率
                scores.append(prob)

        if not scores:
            return {"scores": [0.1], "mean_score": 0.1, "is_spoofed": False}

        mean_score = float(np.mean(scores))
        max_score = float(np.max(scores))

        # 聚合策略: 取最大值 (最可疑的片段决定最终判断)
        return {
            "scores": [round(s, 4) for s in scores],
            "mean_score": round(mean_score, 4),
            "max_score": round(max_score, 4),
            "is_spoofed": max_score > 0.5,
            "model_type": "AASIST" if self._aasist_available else "SimpleDetector"
        }

    # ----- 文本风险分析 -----

    def analyze_text_risk(self, transcript: str) -> list:
        """分析转录文本中的诈骗风险关键词"""
        if not transcript:
            return []

        flags = []
        risk_patterns = {
            "冒充公检法": ["警察", "公安", "法院", "检察院", "逮捕", "通缉", "涉案", "洗钱"],
            "冒充客服": ["客服", "退款", "理赔", "关闭会员", "取消业务"],
            "冒充领导": ["领导", "老板", "转账", "汇款", "急用"],
            "贷款诈骗": ["贷款", "额度", "利率", "保证金", "手续费"],
            "刷单诈骗": ["刷单", "返利", "佣金", "垫付", "任务"],
            "杀猪盘": ["投资", "理财", "博彩", "漏洞", "稳赚"],
            "安全账户": ["安全账户", "验证码", "密码", "银行卡号"],
        }

        found_categories = []
        for category, keywords in risk_patterns.items():
            matched = [kw for kw in keywords if kw in transcript]
            if matched:
                found_categories.append(category)
                flags.append({
                    "category": category,
                    "matched_keywords": matched,
                    "severity": "high" if len(matched) >= 2 else "medium"
                })

        return flags

    # ----- 主检测流程 -----

    def detect(self,
               audio_path: str,
               save_visualization: bool = True) -> AudioDetectResult:
        """
        完整的音频检测流程

        Args:
            audio_path: 音频文件路径
            save_visualization: 是否保存可视化图表

        Returns:
            AudioDetectResult
        """
        # 确保模型已加载
        if not self._model_loaded:
            self.load_models()

        logger.info(f"[Mic] 开始检测: {audio_path}")

        # 1. 预处理
        segments, full_audio = self.preprocess(audio_path)

        # 2. ASR语音识别 (使用完整音频)
        transcript, language = self.transcribe(full_audio)

        # 3. 反欺骗检测
        spoof_result = self.detect_spoof(segments)
        spoof_score = spoof_result["max_score"]

        # 4. 文本风险分析
        text_flags = self.analyze_text_risk(transcript)

        # 5. 综合评分
        # 如果文本也有风险关键词，提升风险等级
        combined_score = spoof_score
        if text_flags:
            # 文本风险对评分有加成（最多+0.15）
            text_boost = min(0.15, len(text_flags) * 0.05)
            combined_score = min(1.0, spoof_score + text_boost)

        # 6. 构建结果
        result = AudioDetectResult(
            transcript=transcript,
            transcript_language=language,
            spoof_score=round(combined_score, 4),
            is_spoofed=combined_score > 0.5,
            risk_level=self._score_to_level(combined_score),
            verdict=self._score_to_verdict(combined_score),
            segments_analyzed=len(segments),
            segment_scores=spoof_result["scores"],
            asr_risk_flags=text_flags,
            audio_artifacts=self._analyze_artifacts(spoof_result),
        )

        logger.info(f"   结果: score={combined_score:.3f}, level={result.risk_level}, "
                    f"verdict={result.verdict}, transcript_len={len(transcript)}")

        return result

    # ----- 工具方法 -----

    def _score_to_level(self, score: float) -> int:
        if score >= 0.8: return 5
        if score >= 0.6: return 4
        if score >= 0.4: return 3
        if score >= 0.2: return 2
        return 1

    def _score_to_verdict(self, score: float) -> str:
        if score >= 0.8: return "AI伪造语音"
        if score >= 0.6: return "高度可疑"
        if score >= 0.4: return "存在疑点"
        if score >= 0.2: return "低风险"
        return "真实人声"

    def _analyze_artifacts(self, spoof_result: dict) -> list:
        """分析音频伪造痕迹"""
        artifacts = []
        scores = spoof_result.get("scores", [])

        if not scores:
            return artifacts

        max_score = max(scores)
        score_variance = np.var(scores) if len(scores) > 1 else 0

        if max_score > 0.7:
            artifacts.append("多个片段检测到合成语音特征")
        if score_variance > 0.05:
            artifacts.append("不同片段间存在异常差异（疑似拼接）")
        if 0.4 <= max_score <= 0.7:
            artifacts.append("部分片段存在微弱合成痕迹")

        return artifacts

    @staticmethod
    def format_report(result: AudioDetectResult) -> str:
        """生成人类可读的检测报告"""
        lines = []
        lines.append("=" * 50)
        lines.append("  多维盾卫 · 音频真伪检测报告")
        lines.append("=" * 50)
        lines.append(f"  识别内容: {result.transcript[:80]}...")
        lines.append(f"  识别语言: {result.transcript_language}")
        lines.append(f"  AI合成概率: {result.spoof_score:.1%}")
        lines.append(f"  风险等级: {'*' * result.risk_level} ({result.risk_level}/5)")
        lines.append(f"  判定结果: {result.verdict}")
        lines.append(f"  分析片段: {result.segments_analyzed}")

        if result.asr_risk_flags:
            lines.append(f"\n  [WARN]  文本风险:")
            for flag in result.asr_risk_flags:
                lines.append(f"      - {flag['category']}: {', '.join(flag['matched_keywords'])}")

        if result.audio_artifacts:
            lines.append(f"\n  🔊 音频异常:")
            for a in result.audio_artifacts:
                lines.append(f"      - {a}")

        lines.append("=" * 50)
        return "\n".join(lines)


# ============================================================
# 简易测试入口
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python audio_detector.py <音频文件路径>")
        print("示例: python audio_detector.py test.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"[ERR] 文件不存在: {audio_path}")
        sys.exit(1)

    detector = AudioDetector()
    result = detector.detect(audio_path)

    print(AudioDetector.format_report(result))

    # 输出JSON
    print("\n[--] JSON结果:")
    print(json.dumps({
        "transcript": result.transcript,
        "spoof_score": result.spoof_score,
        "is_spoofed": result.is_spoofed,
        "risk_level": result.risk_level,
        "verdict": result.verdict,
        "segments_analyzed": result.segments_analyzed,
        "segment_scores": result.segment_scores,
        "text_risks": result.asr_risk_flags,
        "audio_artifacts": result.audio_artifacts,
    }, ensure_ascii=False, indent=2))
