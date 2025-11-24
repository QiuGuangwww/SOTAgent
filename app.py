"""
SotaAgent Gradio Web应用
使用Gradio封装SotaAgent，提供Web界面进行交互
"""
import asyncio
import json
import os
import sys
from typing import Tuple, Optional

import gradio as gr
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types as genai_types

# 添加项目路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入agent
from My_First_Agent.agent import root_agent

APP_NAME = "agents"
SESSION_USER_ID = "web-user"
SESSION_ID = "default-session"

session_service = InMemorySessionService()
runner = Runner(app_name=APP_NAME, agent=root_agent, session_service=session_service)
_session_ready = False
_session_lock = asyncio.Lock()


async def _ensure_runner_session():
    """确保存在用于当前Web会话的ADK Session。"""
    global _session_ready
    if _session_ready:
        return
    async with _session_lock:
        if _session_ready:
            return
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=SESSION_USER_ID,
            session_id=SESSION_ID,
        )
        if session is None:
            await session_service.create_session(
                app_name=APP_NAME,
                user_id=SESSION_USER_ID,
                session_id=SESSION_ID,
            )
        _session_ready = True


async def _reset_runner_session():
    """清理现有Session，便于重新开始对话。"""
    global _session_ready
    try:
        await session_service.delete_session(
            app_name=APP_NAME,
            user_id=SESSION_USER_ID,
            session_id=SESSION_ID,
        )
    except Exception:
        pass
    _session_ready = False


# 创建papers目录（如果不存在）
PAPER_DIR = "papers"
os.makedirs(PAPER_DIR, exist_ok=True)


async def collect_agent_response(message_str: str, filter_mode: str = "strict", use_vision: bool = False, vision_model: str = "gpt-4o") -> list:
    """
    异步收集Agent的响应chunks，通过ADK Runner调用Agent。
    
    Args:
        message_str: 用户消息
        filter_mode: 过滤模式，"strict"（严格）或 "relaxed"（宽松）
    """
    chunks = []
    try:
        await _ensure_runner_session()
        normalized_message = message_str if isinstance(message_str, str) else str(message_str)
        
        # 将过滤模式和 Vision Model 信息附加到消息中
        mode_hint = "\n[系统提示：当前过滤模式为" + ("严格模式" if filter_mode == "strict" else "宽松模式") + "。在调用 get_latest_sota 等工具时，请根据过滤模式决定是否放宽约束条件。宽松模式下，如果严格过滤没有结果，应自动放宽约束返回候选结果。]"
        
        vision_hint = ""
        if use_vision:
            vision_hint = f"\n[系统提示：已启用 Vision Model 增强提取（{vision_model}）。在调用 run_trustworthy_sota_search 时，请传递 use_vision=True 和 vision_model='{vision_model}' 参数以启用 Vision Model 处理复杂表格和图表。]"
        
        enhanced_message = normalized_message.strip() + mode_hint + vision_hint
        
        user_content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=enhanced_message)],
        )

        async for event in runner.run_async(
            user_id=SESSION_USER_ID,
            session_id=SESSION_ID,
            new_message=user_content,
        ):
            chunks.append(event)
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"直接调用模型失败详情:\n{error_detail}")
        raise Exception(
            f"无法调用Agent。\n"
            f"错误: {e}\n\n"
            f"请检查Agent配置和API密钥是否正确。"
        )
    return chunks


def chat_with_agent(message: str, history: list, filter_mode: str = "严格模式", use_vision: bool = False, vision_model: str = "gpt-4o") -> Tuple[str, list]:
    """
    与Agent进行对话
    
    Args:
        message: 用户消息
        history: 对话历史
        filter_mode: 过滤模式，"严格模式" 或 "宽松模式"
    """
    if not message or not message.strip():
        return "", history
    
    history = history or []

    try:
        message_str = message if isinstance(message, str) else (str(message) if message else "")
        
        # 转换过滤模式为内部格式
        internal_mode = "relaxed" if filter_mode == "宽松模式" else "strict"
        
        # Vision Model 参数（从 UI 传入）
        use_vision_flag = use_vision
        vision_model_name = vision_model

        def _content_to_text(content) -> str:
            if not content:
                return ""
            if isinstance(content, str):
                return content.strip()
            parts = getattr(content, "parts", None) or []
            texts = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    texts.append(part_text.strip())
            return "\n".join(texts).strip()

        def _extract_event_text(event) -> str:
            if not hasattr(event, "content") or not getattr(event, "content"):
                return ""
            segments = []
            parts = getattr(event.content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    segments.append(part_text.strip())
                    continue
                func_resp = getattr(part, "function_response", None)
                if func_resp:
                    payload = getattr(func_resp, "response", None)
                    if isinstance(payload, str) and payload.strip():
                        segments.append(payload.strip())
                    elif hasattr(payload, "parts"):
                        nested = []
                        for nested_part in getattr(payload, "parts", None) or []:
                            nested_text = getattr(nested_part, "text", None)
                            if isinstance(nested_text, str) and nested_text.strip():
                                nested.append(nested_text.strip())
                        if nested:
                            segments.append("\n".join(nested))
                    else:
                        try:
                            segments.append(json.dumps(func_resp.model_dump(), ensure_ascii=False))
                        except Exception:
                            segments.append(str(func_resp))
            if not segments and getattr(event, "actions", None):
                delta = getattr(event.actions, "state_delta", None)
                if delta:
                    try:
                        segments.append(json.dumps(delta, ensure_ascii=False, indent=2))
                    except Exception:
                        segments.append(str(delta))
            return "\n".join(seg for seg in segments if seg).strip()

        def _sanitize_agent_output(text: Optional[str]) -> str:
            if not isinstance(text, str):
                return ""
            filtered_lines = []
            banned_keywords = [
                "tool call",
                "toolcall",
                "get_latest_sota",
                "list_common_benchmarks",
                "recent_by_nl",
            ]
            for line in text.splitlines():
                stripped = line.strip()
                lowered = stripped.lower()
                if not stripped:
                    filtered_lines.append("")
                    continue
                if any(keyword in lowered for keyword in banned_keywords):
                    continue
                filtered_lines.append(line)
            cleaned = "\n".join(filtered_lines).strip()
            return cleaned or text.strip()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, collect_agent_response(message_str, internal_mode, use_vision_flag, vision_model_name))
                    chunks = future.result()
            else:
                chunks = loop.run_until_complete(collect_agent_response(message_str, internal_mode, use_vision_flag, vision_model_name))
        except RuntimeError:
            chunks = asyncio.run(collect_agent_response(message_str, internal_mode, use_vision_flag, vision_model_name))

        response = None
        agent_name = getattr(root_agent, "name", None)

        for chunk in reversed(chunks):
            if hasattr(chunk, "author") and agent_name and chunk.author == agent_name:
                event_text = _extract_event_text(chunk)
                if event_text:
                    response = event_text
                    break

        if not response:
            for chunk in chunks:
                if isinstance(chunk, str) and chunk.strip():
                    response = chunk
                    break
                elif hasattr(chunk, 'content'):
                    content_text = _content_to_text(chunk.content)
                    if content_text:
                        response = content_text
                        break
                elif hasattr(chunk, 'text'):
                    text_val = getattr(chunk, "text")
                    if isinstance(text_val, str) and text_val.strip():
                        response = text_val
                        break
        
        if not response:
            response = "Agent已处理请求，但无法提取响应内容。请查看终端日志获取详细信息。"
        
        if not isinstance(response, str):
            try:
                if hasattr(response, '__str__'):
                    response = response.__str__()
                elif hasattr(response, '__repr__'):
                    response = response.__repr__()
                else:
                    response = f"[响应对象: {type(response).__name__}]"
            except Exception as e:
                response = f"[无法转换响应: {str(e)}]"
        
        response = _sanitize_agent_output(response)

        if not response or not response.strip():
            response = "抱歉，我没有理解您的问题。请尝试重新表述您的问题。"
        
        history.append((message, response))
        return "", history
    except Exception as e:
        error_msg = f"❌ 处理请求时出错: {str(e)}\n\n请检查：\n1. API密钥是否正确配置\n2. 网络连接是否正常\n3. 输入的问题是否有效"
        history.append((message, error_msg))
        return "", history


async def clear_chat() -> Tuple[list, str]:
    """清空聊天历史"""
    await _reset_runner_session()
    return [], ""


def _apply_theme(selection: str) -> str:
    """根据选择返回对应的主题脚本"""
    mode = "dark" if selection == "酷炫夜间" else "light"
    return f"<script>document.documentElement.setAttribute('data-theme', '{mode}');</script>"


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    color-scheme: dark;
    --bg-gradient: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.3), transparent),
                   radial-gradient(ellipse 60% 50% at 50% 100%, rgba(168, 85, 247, 0.2), transparent),
                   linear-gradient(180deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    --panel-bg: rgba(15, 23, 42, 0.85);
    --panel-border: rgba(99, 102, 241, 0.4);
    --panel-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(99, 102, 241, 0.1);
    --text-color: #f1f5f9;
    --muted-text: #94a3b8;
    --hero-bg: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15), rgba(14, 165, 233, 0.15));
    --hero-border: rgba(99, 102, 241, 0.5);
    --hero-glow: 0 0 40px rgba(99, 102, 241, 0.3);
    --feature-bg: rgba(30, 41, 59, 0.6);
    --feature-border: rgba(148, 163, 184, 0.2);
    --feature-text: #e2e8f0;
    --card-bg: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.8));
    --card-border: rgba(99, 102, 241, 0.3);
    --accent-gradient: linear-gradient(135deg, #818cf8 0%, #a855f7 50%, #ec4899 100%);
    --accent-shadow: 0 10px 40px rgba(129, 140, 248, 0.4), 0 0 20px rgba(168, 85, 247, 0.3);
    --accent-hover: linear-gradient(135deg, #9ca3f0 0%, #c084fc 50%, #f472b6 100%);
    --input-bg: rgba(30, 41, 59, 0.7);
    --input-border: rgba(148, 163, 184, 0.3);
    --input-focus: rgba(99, 102, 241, 0.5);
    --input-focus-glow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    --secondary-border: rgba(129, 140, 248, 0.5);
    --secondary-text: #c7d2fe;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
}

:root[data-theme="light"] {
    color-scheme: light;
    --bg-gradient: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.12), transparent),
                   radial-gradient(ellipse 60% 50% at 50% 100%, rgba(168, 85, 247, 0.1), transparent),
                   linear-gradient(180deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%);
    --panel-bg: rgba(255, 255, 255, 0.98);
    --panel-border: rgba(99, 102, 241, 0.25);
    --panel-shadow: 0 20px 60px rgba(15, 23, 42, 0.08), 0 0 0 1px rgba(99, 102, 241, 0.08);
    --text-color: #0f172a;
    --muted-text: #475569;
    --hero-bg: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(168, 85, 247, 0.12), rgba(14, 165, 233, 0.12));
    --hero-border: rgba(99, 102, 241, 0.3);
    --hero-glow: 0 0 30px rgba(99, 102, 241, 0.2);
    --feature-bg: rgba(255, 255, 255, 0.95);
    --feature-border: rgba(99, 102, 241, 0.2);
    --feature-text: #1e293b;
    --card-bg: linear-gradient(135deg, rgba(255, 255, 255, 1), rgba(248, 250, 252, 1));
    --card-border: rgba(99, 102, 241, 0.25);
    --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    --accent-shadow: 0 10px 30px rgba(99, 102, 241, 0.25), 0 0 15px rgba(139, 92, 246, 0.15);
    --accent-hover: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #f472b6 100%);
    --input-bg: rgba(255, 255, 255, 1);
    --input-border: rgba(99, 102, 241, 0.3);
    --input-focus: rgba(99, 102, 241, 0.5);
    --input-focus-glow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    --secondary-border: rgba(99, 102, 241, 0.5);
    --secondary-text: #4f46e5;
}

* {
    transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
}

body {
    background: var(--bg-gradient);
    background-attachment: fixed;
    color: var(--text-color);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-weight: 400;
    line-height: 1.6;
    min-height: 100vh;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 2rem 4rem !important;
    border-radius: 0 !important;
    background: var(--panel-bg) !important;
    border: none !important;
    box-shadow: none !important;
    backdrop-filter: blur(20px) saturate(180%);
    position: relative;
    overflow: hidden;
    min-height: 100vh;
}

/* 主内容区域 */
.gradio-row {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    gap: 2rem !important;
}

.gradio-column {
    width: 100% !important;
}

.gradio-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), transparent);
}
.hero {
    background: var(--hero-bg);
    border-radius: 1.5rem;
    border: 1px solid var(--hero-border);
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    color: var(--text-color);
    position: relative;
    overflow: hidden;
    box-shadow: var(--hero-glow);
    backdrop-filter: blur(10px);
}

:root[data-theme="light"] .hero {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(168, 85, 247, 0.1), rgba(14, 165, 233, 0.1));
    border-color: rgba(99, 102, 241, 0.3);
}

.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.2), transparent);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(-20px, -20px) scale(1.1); }
}

.hero .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.2em;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--muted-text);
    margin-bottom: 1rem;
    display: inline-block;
    padding: 0.5rem 1rem;
    background: rgba(99, 102, 241, 0.1);
    border-radius: 2rem;
    border: 1px solid rgba(99, 102, 241, 0.2);
}

.hero h1 {
    font-size: 2.75rem;
    font-weight: 800;
    margin-bottom: 1rem;
    line-height: 1.2;
    color: var(--text-color) !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    -webkit-text-fill-color: var(--text-color) !important;
}

:root[data-theme="light"] .hero h1 {
    color: #0f172a !important;
    text-shadow: 0 2px 8px rgba(99, 102, 241, 0.2);
    -webkit-text-fill-color: #0f172a !important;
}

.hero p {
    color: var(--text-color);
    opacity: 0.85;
    font-size: 1.05rem;
    max-width: 750px;
    line-height: 1.7;
}

:root[data-theme="light"] .hero p {
    color: #334155;
    opacity: 0.9;
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    padding: 0;
    margin-top: 1.5rem;
    list-style: none;
}

.feature-grid li {
    background: var(--feature-bg);
    border: 1px solid var(--feature-border);
    border-radius: 1rem;
    padding: 1.25rem 1.5rem;
    font-size: 0.95rem;
    color: var(--text-color);
    font-weight: 500;
    backdrop-filter: blur(10px);
    cursor: default;
    position: relative;
    overflow: hidden;
}

:root[data-theme="light"] .feature-grid li {
    color: #1e293b;
    background: rgba(255, 255, 255, 0.9);
    border-color: rgba(99, 102, 241, 0.25);
}

.feature-grid li::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.1), transparent);
    transition: left 0.5s ease;
}

.feature-grid li:hover::before {
    left: 100%;
}

.feature-grid li:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.5);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1.25rem;
    margin-bottom: 2rem;
}

.stat-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 1.25rem;
    padding: 1.75rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.stat-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--accent-gradient);
    transform: scaleX(0);
    transition: transform 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    border-color: rgba(99, 102, 241, 0.5);
}

.stat-card:hover::after {
    transform: scaleX(1);
}

.stat-card .stat-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: var(--text-color) !important;
    text-shadow: 0 2px 8px rgba(99, 102, 241, 0.4);
    -webkit-text-fill-color: var(--text-color) !important;
    display: block;
    margin-bottom: 0.5rem;
    line-height: 1;
}

:root[data-theme="light"] .stat-card .stat-value {
    color: #6366f1 !important;
    text-shadow: 0 2px 6px rgba(99, 102, 241, 0.3);
    -webkit-text-fill-color: #6366f1 !important;
}

.stat-card .stat-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-color);
    opacity: 0.9;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.gradio-chatbot {
    width: 100% !important;
    max-width: 100% !important;
    border-radius: 1.5rem !important;
    border: 2px solid var(--panel-border) !important;
    background: var(--input-bg) !important;
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    min-height: 600px !important;
    max-height: 75vh !important;
    padding: 2rem !important;
    overflow-y: auto !important;
}

.gradio-chatbot > div {
    width: 100% !important;
    max-width: 100% !important;
}

:root[data-theme="light"] .gradio-chatbot {
    background: rgba(255, 255, 255, 0.98) !important;
    box-shadow: 0 8px 32px rgba(15, 23, 42, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
    border-color: rgba(99, 102, 241, 0.25) !important;
}

.gradio-chatbot .user {
    background: var(--accent-gradient) !important;
    color: #fff !important;
    border-radius: 1.25rem !important;
    padding: 1.25rem 1.5rem !important;
    margin: 1rem 0 !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    max-width: 75% !important;
    box-shadow: 0 6px 20px rgba(129, 140, 248, 0.4), 0 2px 8px rgba(168, 85, 247, 0.3) !important;
    border: none !important;
    font-weight: 500 !important;
    line-height: 1.6 !important;
    word-wrap: break-word !important;
}

.gradio-chatbot .bot {
    background: var(--feature-bg) !important;
    color: var(--text-color) !important;
    border-radius: 1.25rem !important;
    padding: 1.25rem 1.5rem !important;
    margin: 1rem 0 !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    max-width: 80% !important;
    border: 1px solid var(--feature-border) !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    font-weight: 400 !important;
    line-height: 1.7 !important;
    word-wrap: break-word !important;
}

:root[data-theme="light"] .gradio-chatbot .bot {
    background: rgba(248, 250, 252, 0.95) !important;
    color: #1e293b !important;
    border-color: rgba(99, 102, 241, 0.25) !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
}

.gradio-textbox {
    border-radius: 1rem !important;
    width: 100% !important;
}

.gradio-textbox textarea {
    border-radius: 1rem !important;
    border: 2px solid var(--input-border) !important;
    min-height: 100px !important;
    background: var(--input-bg) !important;
    color: var(--text-color) !important;
    padding: 1.25rem 1.5rem !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
    transition: all 0.3s ease !important;
    backdrop-filter: blur(10px);
    width: 100% !important;
    resize: vertical !important;
}

.gradio-textbox textarea:focus {
    border-color: var(--input-focus) !important;
    box-shadow: var(--input-focus-glow) !important;
    outline: none !important;
    background: var(--input-bg) !important;
    transform: translateY(-2px);
}

.gradio-button {
    border-radius: 0.875rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    padding: 0.75rem 1.5rem !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    position: relative;
    overflow: hidden;
}

.gradio-button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.2);
    transform: translate(-50%, -50%);
    transition: width 0.6s ease, height 0.6s ease;
}

.gradio-button:hover::before {
    width: 300px;
    height: 300px;
}

.gradio-button.primary {
    background: var(--accent-gradient) !important;
    border: none !important;
    box-shadow: var(--accent-shadow) !important;
    color: #fff !important;
}

.gradio-button.primary:hover {
    background: var(--accent-hover) !important;
    transform: translateY(-2px);
    box-shadow: 0 15px 45px rgba(129, 140, 248, 0.5), 0 0 25px rgba(168, 85, 247, 0.4) !important;
}

.gradio-button.secondary {
    border: 2px solid var(--secondary-border) !important;
    color: var(--secondary-text) !important;
    background: transparent !important;
    backdrop-filter: blur(10px);
}

.gradio-button.secondary:hover {
    background: rgba(99, 102, 241, 0.1) !important;
    border-color: rgba(99, 102, 241, 0.6) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.2) !important;
}
.filter-mode-selector {
    background: var(--feature-bg) !important;
    border: 1px solid var(--feature-border) !important;
    border-radius: 1rem !important;
    padding: 1.25rem !important;
    margin-bottom: 1.25rem !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    overflow: hidden !important;
}

.filter-mode-selector > * {
    border-radius: 1rem !important;
    overflow: hidden !important;
}

.filter-mode-selector .gr-radio {
    border-radius: 1rem !important;
    overflow: hidden !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.filter-mode-selector:hover {
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
}

.filter-mode-selector label {
    color: var(--text-color) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.75rem !important;
    display: block;
}

:root[data-theme="light"] .filter-mode-selector label {
    color: #0f172a !important;
}

.filter-mode-selector .gr-radio-group {
    margin-top: 0.75rem !important;
    display: flex;
    gap: 1rem;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    box-shadow: none !important;
}

.filter-mode-selector .gr-radio-group label {
    color: var(--text-color) !important;
    opacity: 0.8;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    border-radius: 0.75rem !important;
    border: 1px solid var(--feature-border) !important;
    background: transparent !important;
    transition: all 0.3s ease !important;
    cursor: pointer;
    margin: 0 !important;
    box-shadow: none !important;
}

:root[data-theme="light"] .filter-mode-selector .gr-radio-group label {
    color: #334155 !important;
    border-color: rgba(99, 102, 241, 0.3) !important;
}

.filter-mode-selector .gr-radio-group label:hover {
    background: rgba(99, 102, 241, 0.1) !important;
    border-color: rgba(99, 102, 241, 0.4) !important;
    opacity: 1;
}

.filter-mode-selector .gr-radio-group input[type="radio"]:checked + span {
    color: var(--secondary-text) !important;
    font-weight: 600 !important;
}

.filter-mode-selector .gr-radio-group input[type="radio"]:checked ~ label {
    background: rgba(99, 102, 241, 0.15) !important;
    border-color: rgba(99, 102, 241, 0.5) !important;
    color: var(--secondary-text) !important;
    opacity: 1;
}

:root[data-theme="light"] .filter-mode-selector .gr-radio-group input[type="radio"]:checked ~ label {
    background: rgba(99, 102, 241, 0.12) !important;
    border-color: rgba(99, 102, 241, 0.4) !important;
    color: #4f46e5 !important;
}

.gradio-accordion {
    border-radius: 1rem !important;
    border: 1px solid var(--feature-border) !important;
    background: var(--feature-bg) !important;
    margin-bottom: 1rem !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    transition: all 0.3s ease;
}

.gradio-accordion:hover {
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
}

.gradio-accordion .gradio-accordion-header {
    background: var(--feature-bg) !important;
    color: var(--text-color) !important;
    padding: 1rem 1.25rem !important;
    border-radius: 1rem 1rem 0 0 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border-bottom: 1px solid var(--feature-border) !important;
}

:root[data-theme="light"] .gradio-accordion .gradio-accordion-header {
    background: rgba(255, 255, 255, 0.95) !important;
    color: #0f172a !important;
    border-bottom-color: rgba(99, 102, 241, 0.2) !important;
}

.gradio-accordion .gradio-accordion-content {
    padding: 1.25rem !important;
    background: var(--feature-bg) !important;
}

.gradio-checkbox {
    margin-bottom: 1rem !important;
}

.gradio-checkbox label {
    color: var(--text-color) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    cursor: pointer;
}

:root[data-theme="light"] .gradio-checkbox label {
    color: #1e293b !important;
}

.gradio-checkbox input[type="checkbox"] {
    width: 18px !important;
    height: 18px !important;
    border-radius: 0.375rem !important;
    border: 2px solid var(--feature-border) !important;
    cursor: pointer;
    transition: all 0.3s ease !important;
}

.gradio-checkbox input[type="checkbox"]:checked {
    background: var(--accent-gradient) !important;
    border-color: transparent !important;
}

.gradio-radio {
    margin-top: 0.75rem !important;
}

.gradio-radio label {
    color: var(--text-color) !important;
    opacity: 0.8;
    font-size: 0.9rem !important;
    padding: 0.5rem 0.75rem !important;
    border-radius: 0.5rem !important;
    transition: all 0.3s ease !important;
}

:root[data-theme="light"] .gradio-radio label {
    color: #334155 !important;
}

.gradio-radio input[type="radio"]:checked + span {
    color: var(--secondary-text) !important;
    font-weight: 600 !important;
    opacity: 1;
}

:root[data-theme="light"] .gradio-radio input[type="radio"]:checked + span {
    color: #4f46e5 !important;
}
.sidebar-card {
    background: var(--feature-bg);
    border-radius: 1.25rem;
    border: 1px solid var(--feature-border);
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.sidebar-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
}

.sidebar-card h3 {
    color: var(--text-color) !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    margin-bottom: 1rem !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

:root[data-theme="light"] .sidebar-card h3 {
    color: #0f172a !important;
}

.prompt-list {
    list-style: none;
    padding: 0;
    margin: 0.75rem 0 0;
}

.prompt-list li {
    padding: 0.6rem 0;
    color: var(--text-color);
    opacity: 0.85;
    font-size: 0.9rem;
    line-height: 1.6;
    border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    transition: all 0.3s ease;
}

:root[data-theme="light"] .prompt-list li {
    color: #334155;
    opacity: 0.9;
    border-bottom-color: rgba(99, 102, 241, 0.15);
}

.prompt-list li:last-child {
    border-bottom: none;
}

.prompt-list li:hover {
    color: var(--text-color);
    opacity: 1;
    padding-left: 0.5rem;
}

:root[data-theme="light"] .prompt-list li:hover {
    color: #0f172a;
}

.footer {
    text-align: center !important;
    margin-top: 2rem !important;
    padding: 1.5rem !important;
    color: var(--text-color) !important;
    opacity: 0.8;
    font-size: 0.9rem;
    border-top: 1px solid var(--feature-border);
}

:root[data-theme="light"] .footer {
    color: #475569 !important;
    border-top-color: rgba(99, 102, 241, 0.2) !important;
}

.gradio-examples {
    border-radius: 1rem !important;
    border: 1px solid var(--feature-border) !important;
    background: var(--feature-bg) !important;
    padding: 1rem !important;
    margin-top: 1rem !important;
}

.gradio-examples .example {
    background: var(--input-bg) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 0.75rem !important;
    padding: 0.75rem 1rem !important;
    margin: 0.5rem 0 !important;
    color: var(--text-color) !important;
    cursor: pointer;
    transition: all 0.3s ease !important;
    font-weight: 500;
}

:root[data-theme="light"] .gradio-examples .example {
    background: rgba(255, 255, 255, 1) !important;
    border-color: rgba(99, 102, 241, 0.3) !important;
    color: #1e293b !important;
}

.gradio-examples .example:hover {
    background: rgba(99, 102, 241, 0.1) !important;
    border-color: rgba(99, 102, 241, 0.4) !important;
    transform: translateX(4px);
    color: var(--text-color) !important;
}

:root[data-theme="light"] .gradio-examples .example:hover {
    background: rgba(99, 102, 241, 0.08) !important;
    border-color: rgba(99, 102, 241, 0.5) !important;
    color: #0f172a !important;
}

/* 滚动条美化 */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--feature-bg);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--accent-gradient);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-hover);
}

/* 加载动画 */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

/* 对话消息气泡优化 */
.gradio-chatbot .message {
    margin-bottom: 1.5rem !important;
}

.gradio-chatbot .message-wrap {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

/* 输入区域优化 */
.gradio-textbox-container {
    width: 100% !important;
    margin-top: 1.5rem !important;
}

/* 按钮组优化 */
.gradio-button-group {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .gradio-container {
        padding: 1.5rem !important;
    }
    
    .hero {
        padding: 2rem 1.5rem !important;
    }
    
    .hero h1 {
        font-size: 2rem !important;
    }
    
    .stat-grid {
        grid-template-columns: 1fr !important;
    }
    
    .feature-grid {
        grid-template-columns: 1fr !important;
    }
    
    .gradio-chatbot {
        min-height: 400px !important;
        padding: 1rem !important;
    }
    
    .gradio-chatbot .user,
    .gradio-chatbot .bot {
        max-width: 90% !important;
        padding: 1rem !important;
    }
}
"""


# 创建Gradio界面
with gr.Blocks(
    title="SotaAgent - SOTA模型查询助手",
    theme=gr.themes.Soft(),
    css=custom_css,
    head="""
    <script>
    // 页面加载时设置初始主题
    (function() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                document.documentElement.setAttribute('data-theme', 'dark');
            });
        } else {
            document.documentElement.setAttribute('data-theme', 'dark');
        }
    })();
    </script>
    """
) as iface:
    with gr.Row():
        theme_toggle = gr.Radio(
            ["酷炫夜间", "简洁浅色"],
            value="酷炫夜间",
            label="界面主题",
        )

    gr.Markdown(
        """
        <div class="hero">
            <p class="eyebrow">SotaAgent · 研究辅助面板</p>
            <h1>精准检索基准 · 秒回最新 SOTA · 中文交互更自然</h1>
            <p>整合 arXiv、Benchmark 配置与自定义工具链，帮助你快速定位实验表格、指标与模型亮点，支持自然语言与参数化双模式。</p>
            <ul class="feature-grid">
                <li>📚 论文检索与摘要速览</li>
                <li>🏆 get_latest_sota 自动调用</li>
                <li>📊 Benchmark 过滤 + 约束</li>
                <li>⚙️ 中文提示词模版内置</li>
        </ul>
        </div>
        <div class="stat-grid">
            <div class="stat-card">
                <span class="stat-value">120+</span>
                <span class="stat-label">覆盖 Benchmarks</span>
            </div>
            <div class="stat-card">
                <span class="stat-value">400+</span>
                <span class="stat-label">可检索论文属性</span>
            </div>
            <div class="stat-card">
                <span class="stat-value">3s</span>
                <span class="stat-label">平均响应时间</span>
            </div>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=8):
            # 获取图片路径（相对于 app.py 的位置）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            avatar_image_path = os.path.join(parent_dir, "人工智能_ 人工智能_ 自动机_ 脑_ 数码产品_ 机器人学_爱给网_aigei_com.png")
            # 如果图片不存在，尝试当前目录
            if not os.path.exists(avatar_image_path):
                avatar_image_path = os.path.join(current_dir, "人工智能_ 人工智能_ 自动机_ 脑_ 数码产品_ 机器人学_爱给网_aigei_com.png")
            # 如果还是不存在，使用默认emoji
            if not os.path.exists(avatar_image_path):
                avatar_image_path = "🤖"
            
            chatbot = gr.Chatbot(
                label="",
                height=600,
                avatar_images=(None, avatar_image_path),
                type="tuples",
                show_copy_button=True,
                container=True,
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="",
                    placeholder="输入您的问题，例如：找 GOT-10k 上最近的纯监督 SOTA",
                    scale=9,
                    lines=3,
                )
                submit_btn = gr.Button("发送 ✨", variant="primary", scale=1, size="lg")
            
            with gr.Row():
                clear_btn = gr.Button("清空对话", variant="secondary")
                examples = gr.Examples(
                    examples=[
                        "找 GOT-10k 上最新的 SOTA 模型",
                        "RT-1 数据集上纯监督的 SOTA",
                        "VLA常用数据集及其对应的SOTA模型",
                        "搜索关于vision transformer的论文",
                        "列出最近关于强化学习的论文",
                    ],
                    inputs=msg,
                    label="示例问题",
                )
        with gr.Column(scale=4):
            # 过滤模式选择器
            filter_mode_radio = gr.Radio(
                choices=["严格模式", "宽松模式"],
                value="严格模式",
                label="🔍 过滤模式",
                info="严格模式：精确匹配所有约束条件；宽松模式：如果严格过滤无结果，自动放宽约束返回候选",
                elem_classes=["filter-mode-selector"]
            )
            
            # Vision Model 选项
            with gr.Accordion("🤖 Vision Model 增强（Beta）", open=False):
                use_vision_checkbox = gr.Checkbox(
                    value=False,
                    label="启用 Vision Model",
                    info="使用 GPT-4V/Claude Vision 处理复杂表格和图表（需要额外 API 调用，成本较高）"
                )
                vision_model_radio = gr.Radio(
                    choices=["gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash-exp"],
                    value="gpt-4o",
                    label="Vision Model 选择",
                    info="选择用于处理复杂表格和图表的 Vision Model",
                    visible=True
                )
                gr.Markdown(
                    """
                    <div style="font-size: 0.85em; color: var(--muted-text); margin-top: 0.5rem;">
                    <strong>💡 使用建议：</strong><br>
                    • 仅在需要处理复杂表格/图表时启用<br>
                    • Vision Model 会增加处理时间和成本<br>
                    • 基础提取已足够处理大多数简单表格<br>
                    • 启用后，在查询时使用"用可信的方式找..."会自动使用 Vision Model
                    </div>
                    """
                )
            
            gr.Markdown(
                """
                <div class="sidebar-card">
                    <h3>🎯 高效提问技巧</h3>
                    <ul class="prompt-list">
                        <li>➤ 描述 Benchmark + 时间窗口：例如 "GOT-10k 最近 180 天 SOTA"。</li>
                        <li>➤ 加上约束：纯监督 / 零样本 / 不含额外数据。</li>
                        <li>➤ 询问论文时附上 arXiv ID（如 2305.00012）。</li>
                        <li>➤ 需要表格输出时附加 "请整理成表格"。</li>
                    </ul>
                </div>
                """
            )
            with gr.Accordion("快捷模版", open=True):
                gr.Markdown(
                    """
                    - **SOTA 查询**：`找 {Benchmark} 近 {时间} 的最新 SOTA，限制 {scope/constraint}`
                    - **论文检索**：`搜索 {主题} 的论文并总结关键贡献`
                    - **指标对比**：`列出 {Benchmark} 最近 5 篇论文及其指标`
                    """
                )
            with gr.Accordion("常见任务", open=False):
                gr.Markdown(
                    """
                    1. 提取实验表格并比较与自身模型差距  
                    2. 列出 VLA / Tracking 常见基准与官方指标  
                    3. 判断论文是否使用额外数据、是否零样本
                    """
                )

    msg.submit(
        fn=chat_with_agent,
        inputs=[msg, chatbot, filter_mode_radio, use_vision_checkbox, vision_model_radio],
        outputs=[msg, chatbot],
    )
    
    submit_btn.click(
        fn=chat_with_agent,
        inputs=[msg, chatbot, filter_mode_radio, use_vision_checkbox, vision_model_radio],
        outputs=[msg, chatbot],
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg],
    )
    
    theme_toggle.change(
        fn=None,
        inputs=theme_toggle,
        outputs=None,
        js="""
        (selection) => {
            const mode = selection === "酷炫夜间" ? "dark" : "light";
            document.documentElement.setAttribute('data-theme', mode);
            return [];
        }
        """
    )

    gr.Markdown(
        """
        <div class="footer">
        <p>💡 <strong>提示</strong>：使用自然语言提问，Agent会自动理解并调用相应的工具。</p>
        <p>⚠️ <strong>注意</strong>：查询SOTA模型可能需要一些时间，如遇速率限制请稍候重试。</p>
        </div>
        """
    )


if __name__ == "__main__":
    import socket
    
    def find_free_port(start_port=50001, max_attempts=10):
        for i in range(max_attempts):
            port = start_port + i
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('0.0.0.0', port))
                    return port
            except OSError:
                continue
        return None
    
    port = find_free_port(50001)
    if port is None:
        print("警告：50001-50010端口都被占用，Gradio将自动选择可用端口")
        port = None
    
    iface.launch(
        server_name='0.0.0.0',
        server_port=port,
        share=False,
        show_error=True,
        favicon_path=None,
        show_api=False,
    )

