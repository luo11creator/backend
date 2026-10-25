"""
============================================================
多维盾卫 — 音频检测模型下载脚本
============================================================

自动下载所需的预训练模型:
  1. AASIST-L (语音真伪检测) — 从 HuggingFace
  2. Whisper base (语音转文字) — 自动通过 openai-whisper 下载

使用方式:
    python download_audio_models.py              # 下载全部
    python download_audio_models.py --whisper-only   # 仅下载Whisper
    python download_audio_models.py --aasist-only    # 仅下载AASIST

模型存放位置:
    backend/models/
    ├── aasist/              # AASIST模型代码 + 权重
    │   ├── AASIST.py        # 模型定义
    │   └── weights/
    │       └── AASIST-L.pth # 预训练权重 (~85MB)
    └── whisper/             # Whisper模型缓存 (由openai-whisper管理)
============================================================
"""

import os
import sys
import argparse
import subprocess
import urllib.request
from pathlib import Path

# ============================================================
# 配置
# ============================================================
BACKEND_DIR = Path(__file__).parent
MODELS_DIR = BACKEND_DIR / "models"
AASIST_DIR = MODELS_DIR / "aasist"
AASIST_WEIGHTS_DIR = AASIST_DIR / "weights"

# AASIST-L 预训练权重 (HuggingFace)
AASIST_HF_URL = "https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST-L/resolve/main/AASIST-L.pth"

# AASIST 官方仓库
AASIST_REPO = "https://github.com/clovaai/aasist.git"


def ensure_dirs():
    """确保目录结构存在"""
    MODELS_DIR.mkdir(exist_ok=True)
    AASIST_DIR.mkdir(exist_ok=True)
    AASIST_WEIGHTS_DIR.mkdir(exist_ok=True)
    print(f"[DIR] 模型目录: {MODELS_DIR}")


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """下载文件，带进度提示"""
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[OK] {desc} 已存在 ({size_mb:.1f} MB): {dest.name}")
        return True

    print(f"[DL]  下载 {desc} ...")
    print(f"   URL: {url}")

    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                if block_num % 50 == 0:
                    print(f"\r   进度: {percent:.0f}% "
                          f"({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)",
                          end="", flush=True)

        urllib.request.urlretrieve(url, dest, progress_hook)
        print(f"\n[OK] {desc} 下载完成 ({dest.stat().st_size/1024/1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n[ERR] 下载失败: {e}")
        if dest.exists():
            dest.unlink()  # 删除不完整文件
        return False


def download_aasist_weights():
    """下载 AASIST-L 预训练权重"""
    print("\n" + "=" * 60)
    print("[PKG] 下载 AASIST-L 预训练权重")
    print("=" * 60)

    ensure_dirs()

    # 权重文件
    weights_path = AASIST_WEIGHTS_DIR / "AASIST-L.pth"
    success = download_file(AASIST_HF_URL, weights_path, "AASIST-L 权重")

    if not success:
        print("\n[WARN]  自动下载失败，请手动下载:")
        print(f"   1. 访问: https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST-L")
        print(f"   2. 下载 AASIST-L.pth")
        print(f"   3. 放到: {weights_path}")
        return False

    return True


def download_aasist_model_code():
    """下载 AASIST 模型定义代码"""
    print("\n" + "=" * 60)
    print("[PKG] 获取 AASIST 模型代码")
    print("=" * 60)

    model_py = AASIST_DIR / "AASIST.py"

    if model_py.exists():
        print(f"[OK] AASIST.py 已存在: {model_py}")
        return True

    # 方式1: 从 GitHub 直接下载单个文件
    raw_url = "https://raw.githubusercontent.com/clovaai/aasist/main/models/AASIST.py"
    print(f"[DL]  从 GitHub 下载 AASIST.py ...")

    try:
        urllib.request.urlretrieve(raw_url, model_py)
        print(f"[OK] AASIST.py 下载完成")
        return True
    except Exception as e:
        print(f"[WARN]  直接下载失败: {e}")

    # 方式2: 克隆整个仓库后复制
    print("[~] 尝试克隆整个 AASIST 仓库 ...")
    try:
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "aasist"
            subprocess.run(
                ["git", "clone", "--depth", "1", AASIST_REPO, str(tmp_path)],
                check=True, capture_output=True, timeout=120
            )
            # 复制 models 目录
            src_models = tmp_path / "models"
            if src_models.exists():
                for f in src_models.iterdir():
                    dest = AASIST_DIR / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)
                        print(f"   [OK] 复制: {f.name}")
            print("[OK] AASIST 模型代码获取完成")
            return True
    except Exception as e2:
        print(f"[ERR] 克隆仓库也失败了: {e2}")
        print("\n[WARN]  请手动操作:")
        print(f"   1. git clone {AASIST_REPO}")
        print(f"   2. 复制 models/ 目录到: {AASIST_DIR}")
        return False


def download_whisper_model():
    """预下载 Whisper 模型"""
    print("\n" + "=" * 60)
    print("[PKG] 预下载 Whisper base 模型 (语音转文字)")
    print("=" * 60)

    try:
        import whisper
        print("[DL]  正在下载 whisper base 模型 (~74MB)...")
        # 首次加载会自动下载
        model = whisper.load_model("base")
        print("[OK] Whisper base 模型就绪")
        del model
        return True
    except ImportError:
        print("[ERR] openai-whisper 未安装")
        print("   请运行: pip install openai-whisper")
        return False
    except Exception as e:
        print(f"[ERR] 下载失败: {e}")
        print("   Whisper模型将在首次启动服务时自动下载")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载音频检测预训练模型")
    parser.add_argument("--whisper-only", action="store_true", help="仅下载Whisper")
    parser.add_argument("--aasist-only", action="store_true", help="仅下载AASIST")
    args = parser.parse_args()

    print("=" * 60)
    print("  多维盾卫 — 音频模型下载工具")
    print("=" * 60)

    all_ok = True

    if not args.aasist_only:
        if not download_whisper_model():
            all_ok = False

    if not args.whisper_only:
        if not download_aasist_weights():
            all_ok = False
        if not download_aasist_model_code():
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("[OK] 全部模型下载完成！")
        print(f"   模型目录: {MODELS_DIR}")
        print(f"   启动服务: python server.py")
    else:
        print("[WARN]  部分下载失败，请检查上面的错误信息")
    print("=" * 60)


if __name__ == "__main__":
    main()
