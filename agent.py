"""
================================================================================
多维盾卫 —— 反诈智能体（Agent）后端模块
================================================================================

面向「大模型与智能体应用赛道」的智能体核心：让国产大模型（DeepSeek）充当"决策大脑"，
把已有的检测能力（文本诈骗识别 / 视频换脸 / 音频克隆 / 多模态融合）封装为**可被
智能体自主调用的工具**，通过 Function Calling（ReAct）完成多步推理与综合研判，
最终给出诈骗类型判定 + 防骗建议。

架构（端云协同）：
    端侧（Android）: 检测模型离线跑，输出特征/分数，隐私数据不出手机
    云端（本模块）: LLM 智能体调度工具 → 语义理解 → 综合研判 → 生成建议

工具列表：
    1. detect_text        文本诈骗检测（本地关键词规则，可换 TextCNN）
    2. search_cases       诈骗案例知识库检索（内置案例库，可换 FAISS 向量检索）
    3. detect_video       视频换脸检测（占位接口，接 EfficientNet）
    4. detect_audio       音频克隆检测（占位接口，接 AASIST）
    5. detect_multimodal  多模态融合检测（占位接口，接 GuardFusion）

用法：
    python agent.py                 # 命令行交互测试（需先设置 DEEPSEEK_API_KEY）
    或在 server.py 中挂载 router：   app.include_router(agent.router)

DeepSeek API：环境变量 DEEPSEEK_API_KEY，或代码里 DEEPSEEK_API_KEY 常量。
接口兼容 OpenAI 格式：https://api.deepseek.com/chat/completions
================================================================================
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "填写使用者自己的deepseekapi")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"          # 便宜、快、国产，符合"鼓励国产大模型"
MAX_TURNS = 4                              # 最多工具调用轮数，防止死循环


# ===========================================================================
# 工具 1：文本诈骗检测（本地关键词规则，可替换为 TextCNN 推理）
# ===========================================================================

FRAUD_KEYWORDS = {
    "冒充公检法": ["警察", "公安", "法院", "检察院", "逮捕", "通缉", "涉案", "洗钱", "安全账户"],
    "冒充客服": ["客服", "退款", "理赔", "关闭会员", "取消业务", "屏幕共享"],
    "冒充领导": ["领导", "老板", "转账", "汇款", "急用", "不方便接电话"],
    "贷款诈骗": ["贷款", "额度", "利率", "保证金", "手续费", "刷流水"],
    "刷单诈骗": ["刷单", "返利", "佣金", "垫付", "任务", "兼职"],
    "杀猪盘": ["投资", "理财", "博彩", "漏洞", "稳赚", "带你赚钱"],
    "裸聊敲诈": ["裸聊", "视频", "截图", "通讯录", "给钱删除"],
}


def detect_text(text: str) -> Dict[str, Any]:
    """文本诈骗检测工具。返回风险分与命中类型。"""
    if not text:
        return {"risk": 0.0, "types": [], "detail": "文本为空"}
    hits = []
    for category, kws in FRAUD_KEYWORDS.items():
        matched = [kw for kw in kws if kw in text]
        if matched:
            hits.append({"type": category, "keywords": matched,
                         "count": len(matched)})
    risk = min(1.0, 0.15 + 0.15 * len(hits) + 0.1 * sum(h["count"] for h in hits))
    return {"risk": round(risk, 3), "types": [h["type"] for h in hits],
            "detail": json.dumps(hits, ensure_ascii=False) if hits else "未命中关键词"}


# ===========================================================================
# 工具 2：诈骗案例知识库检索（内置案例，可替换为 FAISS 向量检索）
# ===========================================================================

FRAUD_CASES = [
    {"type": "冒充公检法", "keywords": ["警察", "公安", "通缉", "安全账户"],
     "desc": "冒充民警称你涉嫌洗钱，要求把钱转入'安全账户'配合调查。",
     "advice": "公检法不会电话办案、不会让你转账；挂断后拨打 96110 核实。"},
    {"type": "冒充客服", "keywords": ["客服", "退款", "关闭会员"],
     "desc": "冒充电商/银行客服，称误开会员或订单异常，诱导共享屏幕转账。",
     "advice": "通过官方 App 内客服核实，绝不共享屏幕、不透露验证码。"},
    {"type": "冒充领导", "keywords": ["领导", "老板", "转账", "汇款"],
     "desc": "盗用领导头像昵称，以'急用钱、不方便接电话'为由要求转账。",
     "advice": "务必当面或视频核实身份，不凭文字指令转账。"},
    {"type": "刷单返利", "keywords": ["刷单", "返利", "佣金", "垫付"],
     "desc": "先小额返利建立信任，再诱导大额垫付后无法提现。",
     "advice": "所有'刷单'都是诈骗，前期返利只是诱饵，立即停止。"},
    {"type": "杀猪盘", "keywords": ["投资", "理财", "稳赚", "带你赚钱"],
     "desc": "以网恋/交友为名建立信任，诱导到虚假平台投资后卷款失联。",
     "advice": "网上认识的人带你投资理财，几乎都是杀猪盘，切勿转账。"},
]


def search_cases(text: str, top_k: int = 2) -> Dict[str, Any]:
    """案例检索工具：按关键词匹配相似诈骗案例。"""
    if not text:
        return {"cases": []}
    scored = []
    for c in FRAUD_CASES:
        score = sum(1 for kw in c["keywords"] if kw in text)
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    return {"cases": [c for _, c in scored[:top_k]]}


# ===========================================================================
# 工具 3/4/5：视频/音频/多模态检测（占位接口，接入真实 detector）
# ===========================================================================

def detect_video(path_or_desc: str = "") -> Dict[str, Any]:
    """视频换脸检测工具。TODO: 接入 EfficientNet 推理，替换 mock 返回。"""
    return {"risk": 0.5, "is_fake": False, "detail": "视频检测占位（接入 EfficientNet 后可返回真实换脸概率）"}


def detect_audio(path_or_desc: str = "") -> Dict[str, Any]:
    """音频克隆检测工具。TODO: 接入 AASIST 推理。"""
    return {"risk": 0.5, "is_spoofed": False, "detail": "音频检测占位（接入 AASIST 后可返回真实合成语音概率）"}


def detect_multimodal(desc: str = "") -> Dict[str, Any]:
    """多模态融合检测工具。TODO: 接入 GuardFusion 融合推理。"""
    return {"risk": 0.5, "inconsistency": 0.0,
            "detail": "多模态融合占位（接入 GuardFusion 后可返回联合真伪分与跨模态不一致分）"}


# ===========================================================================
# 工具注册表（Function Calling 的 tools schema）
# ===========================================================================

TOOLS = [
    {"type": "function", "function": {
        "name": "detect_text",
        "description": "检测一段文本是否为诈骗话术，返回风险分和命中的诈骗类型",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "要检测的文本内容"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "search_cases",
        "description": "从诈骗案例库检索与文本相似的诈骗案例，返回案例描述和防骗建议",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "用于匹配案例的文本"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "detect_multimodal",
        "description": "对视频+音频+文本做多模态融合检测，判断是否为深度伪造或跨模态矛盾",
        "parameters": {"type": "object", "properties": {
            "desc": {"type": "string", "description": "对输入内容的描述"}},
            "required": ["desc"]}}},
]

TOOL_DISPATCH = {
    "detect_text": lambda a: detect_text(a.get("text", "")),
    "search_cases": lambda a: search_cases(a.get("text", "")),
    "detect_video": lambda a: detect_video(a.get("path_or_desc", "")),
    "detect_audio": lambda a: detect_audio(a.get("path_or_desc", "")),
    "detect_multimodal": lambda a: detect_multimodal(a.get("desc", "")),
}

SYSTEM_PROMPT = (
    "你是「多维盾卫」反诈智能体，帮助用户识别电信网络诈骗。"
    "当用户描述一段可疑信息、消息、电话或视频时，你要主动调用工具进行分析："
    "先调用 detect_text 检测文本风险，再调用 search_cases 检索相似案例，"
    "必要时调用 detect_multimodal 判断是否深度伪造或跨模态矛盾。"
    "最后用中文给用户结论：①是否诈骗 ②诈骗类型 ③风险等级(高/中/低) ④3条以内防骗建议。"
    "不要编造工具没返回的结果；若信息不足，如实说明并建议用户补充。"
)


# ===========================================================================
# DeepSeek 调用（OpenAI 兼容）
# ===========================================================================

def _call_llm(messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY 环境变量")
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    resp = requests.post(
        DEEPSEEK_BASE_URL,
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                 "Content-Type": "application/json"},
        json=payload, timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ===========================================================================
# 智能体主循环（ReAct）
# ===========================================================================

def run_agent(user_message: str) -> Dict[str, Any]:
    """智能体入口：多轮工具调用，直到产出最终回答。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}]
    tool_log = []   # 记录调用过的工具与结果，用于可解释性

    for _ in range(MAX_TURNS):
        data = _call_llm(messages, tools=TOOLS)
        choice = data["choices"][0]["message"]

        # 若模型要调用工具
        if choice.get("tool_calls"):
            messages.append(choice)   # assistant 带 tool_calls 的消息
            for tc in choice["tool_calls"]:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"].get("arguments", "{}"))
                fn_result = TOOL_DISPATCH.get(fn_name, lambda a: {"error": "unknown"})(fn_args)
                tool_log.append({"tool": fn_name, "args": fn_args, "result": fn_result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(fn_result, ensure_ascii=False),
                })
            continue

        # 最终回答
        answer = choice.get("content", "")
        return {"answer": answer, "tools_used": tool_log}

    return {"answer": "分析超时，请简化问题后重试。", "tools_used": tool_log}


# ===========================================================================
# 降级模式：无 API Key 时用本地规则给一个可用的回答（保证演示不断线）
# ===========================================================================

def run_agent_fallback(user_message: str) -> Dict[str, Any]:
    t = detect_text(user_message)
    cases = search_cases(user_message)
    types = t["types"]
    risk = t["risk"]
    level = "高" if risk >= 0.6 else "中" if risk >= 0.3 else "低"
    advice = []
    for c in cases.get("cases", []):
        advice.append(c["advice"])
    if not advice:
        advice = ["保持警惕，不轻信、不转账、不透露验证码。", "如有疑问，拨打 96110 反诈专线核实。"]
    answer = (f"【风险等级】{level}\n"
              f"【疑似诈骗类型】{'、'.join(types) if types else '未命中常见诈骗关键词'}\n"
              f"【防骗建议】\n" + "\n".join(f"- {a}" for a in advice[:3]))
    return {"answer": answer, "tools_used": [
        {"tool": "detect_text", "result": t}, {"tool": "search_cases", "result": cases}],
        "fallback": True}


def chat(user_message: str) -> Dict[str, Any]:
    """对外统一入口：有 Key 走 LLM，无 Key 走本地降级。"""
    if not DEEPSEEK_API_KEY:
        return run_agent_fallback(user_message)
    try:
        return run_agent(user_message)
    except Exception as e:
        result = run_agent_fallback(user_message)
        result["llm_error"] = str(e)
        return result


# ===========================================================================
# FastAPI 路由
# ===========================================================================

def build_router():
    try:
        from fastapi import APIRouter
        from pydantic import BaseModel
    except ImportError:
        return None

    class AgentRequest(BaseModel):
        message: str

    router = APIRouter(prefix="/api/v1", tags=["智能体"])

    @router.post("/agent/chat")
    def agent_chat(req: AgentRequest):
        result = chat(req.message)
        return {"detection_type": "agent", "fraud_detected": False, **result}

    return router


router = build_router()


# ===========================================================================
# 命令行自检
# ===========================================================================

if __name__ == "__main__":
    demo = ("我刚收到一条短信，说我涉嫌洗钱，让我把钱转到安全账户配合调查，"
            "还让我下载一个软件共享屏幕，这是诈骗吗？")
    print("=" * 60)
    print("输入:", demo)
    print("=" * 60)
    result = chat(demo)
    print("\n【智能体回答】\n", result["answer"])
    print("\n【调用的工具】")
    for t in result.get("tools_used", []):
        print(f"  - {t['tool']}: {json.dumps(t['result'], ensure_ascii=False)[:120]}")
