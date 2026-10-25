"""
多维盾卫 — Deepfake换脸API (InsightFace)
"""
import os, logging, traceback
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import numpy as np

logger = logging.getLogger("faceswap")
router = APIRouter(prefix="/api/v1")
REPORTS_DIR = Path(__file__).parent / "reports"

_face_swapper = None
_face_analyser = None
_loaded = False

def _init():
    global _face_swapper, _face_analyser, _loaded
    if _loaded: return

    try:
        import insightface
        from insightface.app import FaceAnalysis

        model_root = os.path.expanduser("~/.insightface")
        logger.info(f"InsightFace model root: {model_root}")

        _face_analyser = FaceAnalysis(name='buffalo_l', root=model_root)
        _face_analyser.prepare(ctx_id=-1)
        logger.info("Face analyser ready")

        # 尝试加载inswapper
        sw_path = os.path.join(model_root, "models", "inswapper_128.onnx")
        logger.info(f"Looking for inswapper at: {sw_path}")
        if os.path.exists(sw_path):
            from insightface.model_zoo import get_model
            _face_swapper = get_model(sw_path)
            logger.info("inswapper loaded")
        else:
            logger.warning(f"inswapper NOT FOUND: {sw_path}")

        _loaded = True
    except Exception as e:
        logger.error(f"Init failed: {e}\n{traceback.format_exc()}")
        raise


@router.post("/faceswap")
async def face_swap(source: UploadFile = File(...), target: UploadFile = File(...)):
    try:
        _init()
    except Exception as e:
        raise HTTPException(500, f"模型加载失败: {e}")

    if _face_swapper is None:
        raise HTTPException(500,
            "inswapper_128.onnx未找到. "
            "请放置到 ~/.insightface/models/inswapper_128.onnx")

    try:
        import cv2
        src_bytes = await source.read()
        tgt_bytes = await target.read()
        src = cv2.imdecode(np.frombuffer(src_bytes, np.uint8), cv2.IMREAD_COLOR)
        tgt = cv2.imdecode(np.frombuffer(tgt_bytes, np.uint8), cv2.IMREAD_COLOR)

        if src is None or tgt is None:
            raise HTTPException(400, "图片解码失败")

        src_faces = _face_analyser.get(src)
        tgt_faces = _face_analyser.get(tgt)

        if not src_faces:
            raise HTTPException(400, f"源图未检测到人脸 (size={src.shape})")
        if not tgt_faces:
            raise HTTPException(400, f"目标图未检测到人脸 (size={tgt.shape})")

        logger.info(f"Source faces: {len(src_faces)}, Target faces: {len(tgt_faces)}")
        result = tgt.copy()
        for face in tgt_faces:
            result = _face_swapper.get(result, face, src_faces[0])

        import uuid, tempfile
        # 限制输出尺寸(最大1024px)
        h, w = result.shape[:2]
        max_side = 1024
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            result = cv2.resize(result, (int(w*scale), int(h*scale)))

        _, buf = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 90])
        from fastapi.responses import Response
        return Response(content=buf.tobytes(), media_type="image/jpeg",
                       headers={"Content-Disposition": "attachment; filename=faceswap_result.jpg"})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Swap error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"换脸失败: {e}")


@router.get("/faceswap/status")
def status():
    return {"analyser": _face_analyser is not None,
            "swapper": _face_swapper is not None}
