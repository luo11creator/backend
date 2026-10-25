"""
============================================================
多维盾卫 — 音频检测快速测试脚本
============================================================

测试音频检测模块的各个组件:
  1. 音频预处理
  2. Whisper ASR（如果有音频文件）
  3. AASIST 推理
  4. 可视化生成

使用方式:
    # 使用测试音频
    python test_audio.py

    # 指定音频文件
    python test_audio.py path/to/audio.wav

    # 生成模拟测试报告图（不需要AI模型）
    python test_audio.py --mock-viz
============================================================
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# 添加backend到路径
sys.path.insert(0, str(Path(__file__).parent))
from audio_detector import AudioDetector, AudioDetectResult

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test")


def test_preprocessing():
    """测试音频预处理"""
    print("\n" + "=" * 60)
    print("测试1: 音频预处理模块")
    print("=" * 60)

    try:
        import librosa
        import numpy as np

        # 生成测试信号 (440Hz正弦波 + 噪声)
        sr = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))

        # 保存临时文件
        tmp_path = "test_tmp_audio.wav"
        import soundfile as sf
        sf.write(tmp_path, audio, sr)

        # 测试预处理
        detector = AudioDetector(use_asr=False, use_aasist=False)
        segments, full_audio = detector.preprocess(tmp_path)

        assert len(segments) > 0, "应该有至少1个片段"
        assert len(full_audio) > 0, "完整音频不应为空"

        print(f"[OK] 预处理测试通过")
        print(f"   原始音频: {duration}s → {len(segments)}个片段")
        print(f"   片段大小: {segments[0].shape}")

        # 清理
        os.remove(tmp_path)
        return True
    except Exception as e:
        print(f"[ERR] 预处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_risk_analysis():
    """测试文本风险分析"""
    print("\n" + "=" * 60)
    print("测试2: 文本风险分析")
    print("=" * 60)

    try:
        detector = AudioDetector(use_asr=False, use_aasist=False)

        # 测试诈骗文本
        fraud_text = "我是市公安局的你涉嫌洗钱请把资金转到安全账户"
        flags = detector.analyze_text_risk(fraud_text)
        print(f"[OK] 诈骗文本检测: {len(flags)}个风险类别")
        for f in flags:
            print(f"   - {f['category']}: {f['matched_keywords']}")

        # 测试正常文本
        normal_text = "今天天气真好我们去公园散步吧"
        flags2 = detector.analyze_text_risk(normal_text)
        print(f"[OK] 正常文本: {len(flags2)}个风险标记 (预期0)")

        assert len(flags) > 0, "诈骗文本应检测到风险"
        assert len(flags2) == 0, "正常文本不应有风险"

        print("[OK] 文本风险分析测试通过")
        return True
    except Exception as e:
        print(f"[ERR] 文本分析测试失败: {e}")
        return False


def test_aasist_model():
    """测试AASIST模型加载和推理"""
    print("\n" + "=" * 60)
    print("测试3: AASIST 模型推理")
    print("=" * 60)

    try:
        import numpy as np
        import torch

        from audio_detector import AASISTModel, SEGMENT_SAMPLES

        # 创建模型
        model = AASISTModel()
        print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 测试前向传播
        test_input = torch.randn(2, 1, SEGMENT_SAMPLES)  # batch=2
        with torch.no_grad():
            output = model(test_input)

        assert output.shape == (2, 2), f"输出形状应为(2,2)，实际为{output.shape}"
        probs = torch.softmax(output, dim=1)

        print(f"[OK] 模型推理测试通过")
        print(f"   输入形状: {test_input.shape}")
        print(f"   输出形状: {output.shape}")
        print(f"   Softmax概率: {probs.tolist()}")

        # 测试SincConv
        from audio_detector import SincConv
        sinc = SincConv(64, 251)
        sinc_out = sinc(test_input)
        print(f"   SincConv输出: {sinc_out.shape}")

        return True
    except Exception as e:
        print(f"[ERR] 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_audio(audio_path: str):
    """使用真实音频文件进行完整测试"""
    print("\n" + "=" * 60)
    print(f"测试4: 完整检测流程 — {audio_path}")
    print("=" * 60)

    if not os.path.exists(audio_path):
        print(f"[ERR] 音频文件不存在: {audio_path}")
        return False

    try:
        detector = AudioDetector(use_asr=False)  # 先不启用ASR（需要下载Whisper）
        start = time.time()
        result = detector.detect(audio_path)
        elapsed = time.time() - start

        print(f"\n[OK] 检测完成 (耗时: {elapsed:.1f}s)")
        print(AudioDetector.format_report(result))

        # 生成可视化
        try:
            from audio_visualizer import AudioVisualizer
            viz = AudioVisualizer()
            report_path = f"test_report_{int(time.time())}.png"
            viz.generate_report(audio_path, result, report_path)
            print(f"\n[==] 报告图: {report_path}")
        except Exception as e:
            print(f"[WARN]  可视化跳过: {e}")

        return True
    except Exception as e:
        print(f"[ERR] 检测失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_visualization():
    """生成模拟可视化报告（不需要AI模型也能测试）"""
    print("\n" + "=" * 60)
    print("测试5: 可视化报告生成 (模拟数据)")
    print("=" * 60)

    try:
        import numpy as np
        import soundfile as sf

        from audio_visualizer import AudioVisualizer

        # 生成模拟音频
        sr = 16000
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration))
        # 模拟语音信号
        audio = (0.3 * np.sin(2 * np.pi * 200 * t) +
                 0.2 * np.sin(2 * np.pi * 500 * t) +
                 0.1 * np.sin(2 * np.pi * 1000 * t) +
                 0.05 * np.random.randn(len(t)))
        audio = audio / np.max(np.abs(audio))

        tmp_path = "test_mock_audio.wav"
        sf.write(tmp_path, audio, sr)

        # 模拟检测结果
        mock_result = {
            "transcript": "喂你好我是快递公司的你的包裹被海关扣留了需要缴纳保证金才能放行请尽快转账到指定账户",
            "spoof_score": 0.87,
            "risk_level": 5,
            "verdict": "AI伪造语音",
            "segment_scores": [0.82, 0.91, 0.78, 0.85, 0.93],
            "asr_risk_flags": [
                {"category": "冒充客服", "matched_keywords": ["快递", "包裹", "海关", "保证金"], "severity": "high"},
                {"category": "冒充公检法", "matched_keywords": ["转账", "账户"], "severity": "medium"}
            ],
            "audio_artifacts": [
                "多个片段检测到合成语音特征",
                "频谱分析发现异常高频分量",
                "不同片段间存在拼接痕迹"
            ]
        }

        viz = AudioVisualizer()
        output_path = "test_audio_report_mock.png"
        viz.generate_report(tmp_path, mock_result, output_path)

        print(f"[OK] 模拟报告图已生成: {output_path}")
        print(f"   文件大小: {os.path.getsize(output_path) / 1024:.1f} KB")

        # 生成独立图表
        viz.quick_gauge(0.87, "test_gauge.png")
        print(f"[OK] 仪表盘: test_gauge.png")

        # 清理
        os.remove(tmp_path)
        return True
    except Exception as e:
        print(f"[ERR] 可视化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="多维盾卫音频检测测试")
    parser.add_argument("audio_file", nargs="?", help="音频文件路径")
    parser.add_argument("--mock-viz", action="store_true", help="生成模拟可视化")
    parser.add_argument("--skip-model", action="store_true", help="跳过模型测试")
    args = parser.parse_args()

    print("=" * 60)
    print("  多维盾卫 · 音频检测模块测试")
    print("=" * 60)

    results = {}

    # 测试1: 预处理
    results["预处理"] = test_preprocessing()

    # 测试2: 文本分析
    results["文本分析"] = test_text_risk_analysis()

    # 测试3: 模型（可选跳过）
    if not args.skip_model:
        results["AASIST模型"] = test_aasist_model()

    # 测试4: 完整流程（如果有音频文件）
    if args.audio_file:
        results["完整检测"] = test_with_real_audio(args.audio_file)

    # 测试5: 可视化
    if args.mock_viz:
        results["可视化"] = test_mock_visualization()
    elif not args.audio_file:
        # 至少跑一下可视化
        results["可视化"] = test_mock_visualization()

    # 总结
    print("\n" + "=" * 60)
    print("[--] 测试总结")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"   {'[OK]' if ok else '[ERR]'} {name}")
    print(f"\n   通过: {passed}/{total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
