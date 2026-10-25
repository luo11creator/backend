"""
============================================================
多维盾卫 — 音频检测可视化引擎
============================================================

为检测结果生成可视化分析图表:
  1. 波形图 + 风险区域标注
  2. 频谱图 (Mel Spectrogram)
  3. 风险仪表盘 (Gauge)
  4. 综合报告图 (组合输出)

使用方式:
    from audio_visualizer import AudioVisualizer
    viz = AudioVisualizer()
    viz.generate_report(audio_path, result, output_path)

适配Android端:
    - 所有图片默认 1080×1920 (竖屏)
    - 中文字体支持
    - 颜色映射: 绿(<40%)/黄(40-70%)/红(>70%)
============================================================
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # 非交互后端 (服务器环境)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AudioVisualizer")

# ============================================================
# 配置
# ============================================================
# 中文字体 — 按优先级查找
CN_FONT_CANDIDATES = [
    "SimHei", "Microsoft YaHei", "PingFang SC",
    "Noto Sans CJK SC", "WenQuanYi Micro Hei",
    "Arial Unicode MS", "DejaVu Sans", "sans-serif"
]

# 配色方案
COLOR_SAFE = "#00C853"       # 安全 绿色
COLOR_LOW = "#64DD17"        # 低风险 浅绿
COLOR_MEDIUM = "#FFD600"     # 中风险 黄色
COLOR_HIGH = "#FF6D00"       # 高风险 橙色
COLOR_DANGER = "#D50000"     # 危险 红色
COLOR_BG = "#0D1117"         # 深色背景
COLOR_CARD_BG = "#161B22"    # 卡片背景
COLOR_TEXT = "#C9D1D9"       # 文字
COLOR_TEXT_SECONDARY = "#8B949E"  # 次要文字
COLOR_ACCENT = "#1A237E"     # 主题色 (多维盾卫蓝)


def find_chinese_font() -> str:
    """查找可用的中文字体"""
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}

    for font in CN_FONT_CANDIDATES:
        if font in available:
            return font

    # Fallback: 使用sans-serif
    logger.warning("未找到中文字体，图表中文可能显示为方块")
    return "sans-serif"


class AudioVisualizer:
    """音频检测可视化器"""

    SAMPLE_RATE = 16000
    # 输出尺寸 (竖屏，适配手机)
    FIG_W = 10.8   # 英寸 → 1080px @ 100dpi
    FIG_H = 19.2   # 英寸 → 1920px @ 100dpi
    DPI = 100

    def __init__(self):
        self.font_name = find_chinese_font()
        self._setup_style()

    def _setup_style(self):
        """配置matplotlib风格"""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': [self.font_name, 'DejaVu Sans', 'sans-serif'],
            'axes.unicode_minus': False,
            'figure.dpi': self.DPI,
            'savefig.dpi': self.DPI,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
        })

    # ================================================================
    # 颜色映射
    # ================================================================

    @staticmethod
    def score_to_color(score: float) -> str:
        """风险分数 → 颜色"""
        if score >= 0.8: return COLOR_DANGER
        if score >= 0.6: return COLOR_HIGH
        if score >= 0.4: return COLOR_MEDIUM
        if score >= 0.2: return COLOR_LOW
        return COLOR_SAFE

    @staticmethod
    def score_to_alpha(score: float) -> float:
        """风险分数 → 热力图透明度"""
        return min(0.6, 0.1 + score * 0.5)

    # ================================================================
    # 组件1: 波形图
    # ================================================================

    def draw_waveform(self,
                      audio: np.ndarray,
                      segment_scores: list = None,
                      ax: plt.Axes = None) -> plt.Axes:
        """
        绘制波形图，高风险区域用红色半透明标注

        Args:
            audio: 波形数据
            segment_scores: 每个片段的AI合成分数
            ax: matplotlib axes (可选)
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))

        duration = len(audio) / self.SAMPLE_RATE
        time_axis = np.linspace(0, duration, len(audio))

        # 波形主体
        ax.plot(time_axis, audio, color='#4FC3F7', linewidth=0.5, alpha=0.8)
        ax.fill_between(time_axis, audio, alpha=0.15, color='#4FC3F7')

        # 高亮风险区域
        if segment_scores:
            seg_duration = 4.0  # 每段4秒
            for i, score in enumerate(segment_scores):
                if score >= 0.4:  # 只标注中高风险
                    start_t = i * seg_duration
                    end_t = min(start_t + seg_duration, duration)
                    color = self.score_to_color(score)
                    ax.axvspan(start_t, end_t, alpha=self.score_to_alpha(score),
                              color=color, label=f'段{i+1}: {score:.0%}' if i == 0 else "")

        # 样式
        ax.set_xlim(0, duration)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel('时间 (秒)', color=COLOR_TEXT_SECONDARY, fontsize=10)
        ax.set_ylabel('振幅', color=COLOR_TEXT_SECONDARY, fontsize=10)
        ax.set_title('音频波形', color=COLOR_TEXT, fontsize=14, fontweight='bold')
        ax.set_facecolor(COLOR_CARD_BG)
        ax.tick_params(colors=COLOR_TEXT_SECONDARY, labelsize=9)
        ax.grid(True, alpha=0.15, color='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(COLOR_TEXT_SECONDARY)
        ax.spines['left'].set_color(COLOR_TEXT_SECONDARY)

        return ax

    # ================================================================
    # 组件2: 频谱图 (Mel Spectrogram)
    # ================================================================

    def draw_spectrogram(self,
                         audio: np.ndarray,
                         ax: plt.Axes = None) -> plt.Axes:
        """绘制Mel频谱图"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        # 计算Mel频谱
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.SAMPLE_RATE,
            n_fft=2048, hop_length=512,
            n_mels=128, fmax=8000
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 绘制
        librosa.display.specshow(
            mel_db, sr=self.SAMPLE_RATE, hop_length=512,
            x_axis='time', y_axis='mel',
            cmap='magma', ax=ax
        )

        ax.set_title('Mel 频谱图', color=COLOR_TEXT, fontsize=14, fontweight='bold')
        ax.set_xlabel('时间', color=COLOR_TEXT_SECONDARY, fontsize=10)
        ax.set_ylabel('频率 (Hz)', color=COLOR_TEXT_SECONDARY, fontsize=10)
        ax.tick_params(colors=COLOR_TEXT_SECONDARY, labelsize=9)

        return ax

    # ================================================================
    # 组件3: 风险仪表盘
    # ================================================================

    def draw_gauge(self,
                   score: float,
                   ax: plt.Axes = None,
                   size: str = "large") -> plt.Axes:
        """
        绘制半圆形风险仪表盘

        Args:
            score: 风险分数 (0~1)
            ax: matplotlib axes
            size: "large" (含详细标注) 或 "compact"
        """
        if ax is None:
            fig_size = (6, 5) if size == "large" else (4, 3.5)
            _, ax = plt.subplots(figsize=fig_size, subplot_kw={'projection': 'polar'})

        # 仪表盘参数
        theta_min, theta_max = np.pi, 0  # 半圆 (从π到0，即从右到左)
        levels = 5
        level_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        level_colors = [COLOR_SAFE, COLOR_LOW, COLOR_MEDIUM, COLOR_HIGH, COLOR_DANGER]
        level_labels = ['安全', '低风险', '可疑', '高风险', 'AI伪造']

        # 绘制弧形段
        for i, ((low, high), color) in enumerate(zip(level_ranges, level_colors)):
            theta = np.linspace(
                theta_min + (theta_max - theta_min) * low,
                theta_min + (theta_max - theta_min) * high,
                50
            )
            r_inner = np.full_like(theta, 1.0)
            r_outer = np.full_like(theta, 1.8)
            ax.fill_between(theta, r_inner, r_outer, color=color, alpha=0.7)

        # 指针
        needle_angle = theta_min + (theta_max - theta_min) * score
        ax.annotate('', xy=(needle_angle, 1.6), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='white',
                                   lw=3, shrinkA=0, shrinkB=0))

        # 中心分数
        color = self.score_to_color(score)
        ax.text(0, 0, f'{score:.0%}', ha='center', va='center',
                fontsize=28 if size == "large" else 22,
                fontweight='bold', color=color)
        ax.text(0, -0.4, f'{self._score_label(score)}', ha='center', va='center',
                fontsize=12 if size == "large" else 10,
                color=color, fontweight='bold')

        # 设置
        ax.set_ylim(0, 2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines.clear()
        ax.set_facecolor('none')

        # 刻度标签
        if size == "large":
            for i, label in enumerate(level_labels):
                angle = theta_min + (theta_max - theta_min) * (i * 0.2 + 0.1)
                ax.text(angle, 0.6, label, ha='center', va='center',
                       fontsize=7, color=COLOR_TEXT_SECONDARY, alpha=0.7)

        return ax

    @staticmethod
    def _score_label(score: float) -> str:
        if score >= 0.8: return "AI伪造"
        if score >= 0.6: return "高风险"
        if score >= 0.4: return "可疑"
        if score >= 0.2: return "低风险"
        return "安全"

    # ================================================================
    # 组件4: 片段分析条
    # ================================================================

    def draw_segment_bars(self,
                          segment_scores: list,
                          ax: plt.Axes = None) -> plt.Axes:
        """绘制片段级分析柱状图"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 2.5))

        n = len(segment_scores)
        if n == 0:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                   transform=ax.transAxes, color=COLOR_TEXT_SECONDARY)
            return ax

        x = range(n)
        colors = [self.score_to_color(s) for s in segment_scores]
        bars = ax.bar(x, segment_scores, color=colors, alpha=0.85, width=0.6)

        # 阈值线
        ax.axhline(y=0.5, color=COLOR_MEDIUM, linestyle='--', linewidth=1.5,
                  label='判定阈值 (0.5)', alpha=0.7)
        ax.axhline(y=0.8, color=COLOR_DANGER, linestyle=':', linewidth=1,
                  label='高危阈值 (0.8)', alpha=0.5)

        # 在每个柱上标分数
        for bar, score in zip(bars, segment_scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                   f'{score:.0%}', ha='center', va='bottom',
                   fontsize=8, color=self.score_to_color(score), fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([f'段{i+1}' for i in x])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('AI合成概率', color=COLOR_TEXT_SECONDARY, fontsize=10)
        ax.set_title('逐段分析', color=COLOR_TEXT, fontsize=14, fontweight='bold')
        ax.set_facecolor(COLOR_CARD_BG)
        ax.tick_params(colors=COLOR_TEXT_SECONDARY, labelsize=9)
        ax.legend(loc='upper right', fontsize=8,
                 facecolor=COLOR_CARD_BG, edgecolor='none',
                 labelcolor=COLOR_TEXT_SECONDARY)
        ax.grid(True, alpha=0.1, color='white', axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(COLOR_TEXT_SECONDARY)
        ax.spines['left'].set_color(COLOR_TEXT_SECONDARY)

        return ax

    # ================================================================
    # 综合报告图 (主输出)
    # ================================================================

    def generate_report(self,
                        audio_path: str,
                        result: dict,
                        output_path: str,
                        dark_mode: bool = True) -> str:
        """
        生成完整的检测报告图 (竖屏布局，适配手机查看)

        布局:
        ┌─────────────────┐
        │   标题 + 日期    │
        ├─────────────────┤
        │  识别文字(ASR)  │
        ├─────────────────┤
        │   风险仪表盘     │
        ├─────────────────┤
        │   音频波形图     │
        ├─────────────────┤
        │   片段分析条     │
        ├─────────────────┤
        │   详细信息卡片   │
        └─────────────────┘

        Args:
            audio_path: 原始音频文件路径
            result: 检测结果 (AudioDetectResult 或 dict)
            output_path: 输出PNG路径
            dark_mode: 深色模式

        Returns:
            output_path
        """
        # 加载音频数据
        audio, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # 提取结果字段
        if hasattr(result, '__dataclass_fields__'):
            transcript = result.transcript
            spoof_score = result.spoof_score
            risk_level = result.risk_level
            verdict = result.verdict
            segment_scores = result.segment_scores
            asr_flags = result.asr_risk_flags
            artifacts = result.audio_artifacts
        else:
            transcript = result.get('transcript', '')
            spoof_score = result.get('spoof_score', 0)
            risk_level = result.get('risk_level', 1)
            verdict = result.get('verdict', '未知')
            segment_scores = result.get('segment_scores', [])
            asr_flags = result.get('asr_risk_flags', [])
            artifacts = result.get('audio_artifacts', [])

        # 颜色主题
        if dark_mode:
            bg = COLOR_BG
            card_bg = COLOR_CARD_BG
            text = COLOR_TEXT
            text_sec = COLOR_TEXT_SECONDARY
        else:
            bg = 'white'
            card_bg = '#F6F8FA'
            text = '#24292F'
            text_sec = '#656D76'

        score_color = self.score_to_color(spoof_score)

        # 创建图形
        fig = plt.figure(figsize=(self.FIG_W, self.FIG_H), facecolor=bg)
        gs = fig.add_gridspec(7, 1, hspace=0.3,
                              height_ratios=[0.4, 0.8, 1.5, 1.2, 1.0, 1.5, 0.5])

        # ---- 1. 标题栏 ----
        ax_title = fig.add_subplot(gs[0], facecolor=bg)
        ax_title.axis('off')
        from datetime import datetime
        title_text = f"多维盾卫 · 音频真伪检测报告"
        date_text = datetime.now().strftime("%Y-%m-%d %H:%M")
        ax_title.text(0.5, 0.7, title_text, transform=ax_title.transAxes,
                     ha='center', va='center', fontsize=22, fontweight='bold',
                     color=COLOR_ACCENT if not dark_mode else '#58A6FF')
        ax_title.text(0.5, 0.3, date_text, transform=ax_title.transAxes,
                     ha='center', va='center', fontsize=11, color=text_sec)

        # ---- 2. ASR识别文字 ----
        ax_asr = fig.add_subplot(gs[1], facecolor=card_bg)
        ax_asr.axis('off')
        # 卡片背景
        self._add_card_background(ax_asr, card_bg, bg)

        if transcript:
            # 截断过长文本
            display_text = transcript[:200] + "..." if len(transcript) > 200 else transcript
            ax_asr.text(0.05, 0.85, "[ASR] 语音识别内容", transform=ax_asr.transAxes,
                       fontsize=14, fontweight='bold', color=text, va='top')
            ax_asr.text(0.05, 0.55, f'"{display_text}"', transform=ax_asr.transAxes,
                       fontsize=12, color=text, va='center',
                       style='italic',
                       bbox=dict(boxstyle='round,pad=0.8', facecolor=bg,
                                edgecolor=text_sec, alpha=0.5))
            # 风险标记
            if asr_flags:
                flags_str = "[WARN] 文本风险: " + ", ".join(
                    f['category'] for f in asr_flags)
                ax_asr.text(0.05, 0.15, flags_str, transform=ax_asr.transAxes,
                           fontsize=10, color=COLOR_HIGH, va='center')
        else:
            ax_asr.text(0.5, 0.5, "（未启用语音识别或识别失败）",
                       transform=ax_asr.transAxes, ha='center', va='center',
                       fontsize=13, color=text_sec)

        # ---- 3. 仪表盘 + 核心指标 ----
        ax_gauge_box = fig.add_subplot(gs[2], facecolor=card_bg)
        ax_gauge_box.axis('off')
        self._add_card_background(ax_gauge_box, card_bg, bg)

        # 左侧: 仪表盘
        ax_gauge = fig.add_axes([0.08, 0.47, 0.42, 0.2], projection='polar')
        self.draw_gauge(spoof_score, ax=ax_gauge, size='large')

        # 右侧: 核心指标
        ax_metrics = fig.add_axes([0.55, 0.48, 0.4, 0.16])
        ax_metrics.axis('off')
        ax_metrics.set_facecolor('none')

        verdict_color = score_color
        metrics = [
            ("判定结果", verdict, verdict_color, 16),
            ("AI合成概率", f"{spoof_score:.1%}", verdict_color, 28),
            (f"风险等级", f"{'★' * risk_level}{'☆' * (5 - risk_level)}", verdict_color, 14),
            ("分析片段", f"{len(segment_scores)} 个", text, 12),
        ]
        for j, (label, value, val_color, fs) in enumerate(metrics):
            y_pos = 0.9 - j * 0.22
            ax_metrics.text(0, y_pos, label, transform=ax_metrics.transAxes,
                          fontsize=11, color=text_sec, va='center')
            ax_metrics.text(0.55, y_pos, value, transform=ax_metrics.transAxes,
                          fontsize=fs, color=val_color, va='center', fontweight='bold')

        # ---- 4. 波形图 ----
        ax_wave = fig.add_subplot(gs[3], facecolor=card_bg)
        self.draw_waveform(audio, segment_scores, ax=ax_wave)
        ax_wave.set_facecolor(card_bg)

        # ---- 5. 片段分析 ----
        ax_bars = fig.add_subplot(gs[4], facecolor=card_bg)
        self.draw_segment_bars(segment_scores, ax=ax_bars)
        ax_bars.set_facecolor(card_bg)

        # ---- 6. 详细分析 ----
        ax_detail = fig.add_subplot(gs[5], facecolor=card_bg)
        ax_detail.axis('off')
        self._add_card_background(ax_detail, card_bg, bg)

        detail_y = 0.9
        ax_detail.text(0.05, detail_y, "[+] 详细分析", transform=ax_detail.transAxes,
                      fontsize=14, fontweight='bold', color=text)

        # 音频异常
        detail_y -= 0.12
        if artifacts:
            ax_detail.text(0.05, detail_y, "音频层面:", transform=ax_detail.transAxes,
                          fontsize=11, color=text_sec, fontweight='bold')
            for art in artifacts[:5]:
                detail_y -= 0.08
                ax_detail.text(0.08, detail_y, f"• {art}", transform=ax_detail.transAxes,
                              fontsize=10, color=text)
        else:
            ax_detail.text(0.05, detail_y, "音频层面: 未发现明显伪造痕迹",
                          transform=ax_detail.transAxes, fontsize=11, color=text)

        # 文本风险
        detail_y -= 0.12
        if asr_flags:
            ax_detail.text(0.05, detail_y, "文本层面:", transform=ax_detail.transAxes,
                          fontsize=11, color=text_sec, fontweight='bold')
            for flag in asr_flags[:5]:
                detail_y -= 0.08
                kws = ", ".join(flag.get('matched_keywords', [])[:3])
                ax_detail.text(0.08, detail_y,
                              f"• {flag['category']}: 匹配 [{kws}]",
                              transform=ax_detail.transAxes,
                              fontsize=10, color=COLOR_HIGH)
        elif transcript:
            detail_y -= 0.08
            ax_detail.text(0.05, detail_y, "文本层面: 未检测到诈骗关键词",
                          transform=ax_detail.transAxes, fontsize=11, color=text)

        # 建议
        detail_y -= 0.15
        suggestions = self._get_suggestions(risk_level)
        ax_detail.text(0.05, detail_y, "[!] 安全建议:", transform=ax_detail.transAxes,
                      fontsize=11, color=text_sec, fontweight='bold')
        for sug in suggestions[:4]:
            detail_y -= 0.08
            ax_detail.text(0.08, detail_y, f"• {sug}", transform=ax_detail.transAxes,
                          fontsize=10, color=text)

        # ---- 7. 页脚 ----
        ax_footer = fig.add_subplot(gs[6], facecolor=bg)
        ax_footer.axis('off')
        ax_footer.text(0.5, 0.5, "多维盾卫 · AI反诈智能检测系统 · 仅供参考",
                      transform=ax_footer.transAxes, ha='center', va='center',
                      fontsize=9, color=text_sec, alpha=0.5)

        # 保存
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                   exist_ok=True)
        fig.savefig(output_path, dpi=self.DPI, facecolor=bg, edgecolor='none',
                    bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)

        logger.info(f"[==] 报告图已保存: {output_path}")
        return output_path

    @staticmethod
    def _add_card_background(ax: plt.Axes, card_color: str, bg_color: str):
        """为Axes添加圆角卡片背景效果"""
        ax.set_facecolor(card_color)
        # Note: matplotlib不原生支持圆角，这里用矩形近似
        for spine in ax.spines.values():
            spine.set_edgecolor(bg_color)
            spine.set_linewidth(0.5)

    @staticmethod
    def _get_suggestions(risk_level: int) -> list:
        if risk_level >= 4:
            return [
                "立即停止与对方的所有通讯",
                "切勿进行任何形式的转账汇款",
                "保存此检测报告作为报警证据",
                "拨打 96110 反诈专线举报"
            ]
        elif risk_level == 3:
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

    # ================================================================
    # 快速单图生成 (轻量版，适合生成单独的图表)
    # ================================================================

    def quick_waveform(self, audio_path: str, output_path: str) -> str:
        """生成独立波形图"""
        audio, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)
        fig, ax = plt.subplots(figsize=(8, 2.5), facecolor=COLOR_BG)
        self.draw_waveform(audio, ax=ax)
        fig.savefig(output_path, facecolor=COLOR_BG)
        plt.close(fig)
        return output_path

    def quick_spectrogram(self, audio_path: str, output_path: str) -> str:
        """生成独立频谱图"""
        audio, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)
        fig, ax = plt.subplots(figsize=(8, 3), facecolor=COLOR_BG)
        self.draw_spectrogram(audio, ax=ax)
        fig.savefig(output_path, facecolor=COLOR_BG)
        plt.close(fig)
        return output_path

    def quick_gauge(self, score: float, output_path: str) -> str:
        """生成独立仪表盘"""
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': 'polar'},
                              facecolor=COLOR_BG)
        self.draw_gauge(score, ax=ax, size='compact')
        fig.savefig(output_path, facecolor=COLOR_BG, transparent=True)
        plt.close(fig)
        return output_path


# ================================================================
# 测试入口
# ================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python audio_visualizer.py <音频文件路径> [输出路径]")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "audio_report.png"

    viz = AudioVisualizer()

    # 模拟检测结果用于测试
    mock_result = {
        "transcript": "你好我是市公安局的你的身份证涉嫌洗钱案件请配合调查",
        "spoof_score": 0.82,
        "risk_level": 5,
        "verdict": "AI伪造语音",
        "segment_scores": [0.78, 0.85, 0.91, 0.72, 0.88],
        "asr_risk_flags": [
            {"category": "冒充公检法", "matched_keywords": ["警察", "公安", "洗钱"], "severity": "high"}
        ],
        "audio_artifacts": [
            "多个片段检测到合成语音特征",
            "不同片段间存在异常差异（疑似拼接）"
        ]
    }

    viz.generate_report(audio_path, mock_result, output_path)
    print(f"[OK] 测试报告图: {output_path}")
