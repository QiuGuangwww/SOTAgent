"""
SotaAgent Gradio Web应用（单一浅色主题 / 全宽布局）
"""
import asyncio
import json
import os
import sys
import time
from typing import Optional

import gradio as gr
import requests
import random
try:
    from google.genai import types as genai_types  # type: ignore
except Exception as _genai_err:
    print(f"[Warn] 导入 google.genai 失败，将使用简易消息包装: {_genai_err}")
    class _FallbackPart:
        def __init__(self, text: str):
            self.text = text
    class _FallbackContent:
        def __init__(self, role: str, parts):
            self.role = role
            self.parts = parts
    class genai_types:  # type: ignore
        Content = _FallbackContent
        Part = _FallbackPart

# 尝试导入 ADK Runner，失败则回退到简易 Runner 实现
USE_ADK = True
try:
    from google.adk.runners import Runner  # type: ignore
    from google.adk.sessions.in_memory_session_service import InMemorySessionService  # type: ignore
except Exception as _adk_import_err:
    print(f"[Warn] 导入 google.adk 失败，将使用简易 Runner：{_adk_import_err}")
    USE_ADK = False
    InMemorySessionService = None  # type: ignore

try:
    from google.adk.models.lite_llm import LiteLlm  # type: ignore
except Exception as _lite_import_err:
    LiteLlm = None  # type: ignore
    print(f"[Warn] 无法导入 LiteLlm: {_lite_import_err}. 将跳过动态模型切换。")
try:
    from google.adk.models.lite_llm import LiteLlm  # type: ignore
except Exception as _lite_import_err:
    LiteLlm = None  # type: ignore
    print(f"[Warn] 无法导入 LiteLlm: {_lite_import_err}. 将跳过动态模型切换。")

# 添加项目路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入agent
from My_First_Agent.agent import root_agent
try:
    from My_First_Agent.agent import PIPELINE_AVAILABLE
except ImportError:
    PIPELINE_AVAILABLE = False

# 兼容补丁：绕过 Gradio 在生成 API Schema 时对布尔 Schema 的处理异常
# 报错：TypeError: argument of type 'bool' is not iterable（来源 gradio_client.utils.get_type）
try:
    from gradio_client import utils as _gradio_client_utils  # type: ignore

    _orig_get_type = getattr(_gradio_client_utils, "get_type", None)
    _orig_json_to_py = getattr(_gradio_client_utils, "_json_schema_to_python_type", None)

    if callable(_orig_get_type):
        def _safe_get_type(schema):  # type: ignore
            if isinstance(schema, bool):
                return "any"
            try:
                return _orig_get_type(schema)  # type: ignore
            except Exception:
                return "any"

        _gradio_client_utils.get_type = _safe_get_type  # type: ignore

    if callable(_orig_json_to_py):
        def _safe_json_schema_to_python_type(schema, defs=None):  # type: ignore
            if isinstance(schema, bool):
                return "any"
            try:
                return _orig_json_to_py(schema, defs)  # type: ignore
            except Exception:
                return "any"

        _gradio_client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type  # type: ignore
except Exception as _patch_err:
    print(f"[Gradio-Compat] Schema 兼容补丁加载失败：{_patch_err}")

def charge_photon(event_value, sku_id, request: gr.Request):
    """
    光子扣费接口
    """
    # 优先取 Cookie 中的 accessKey
    cookies = request.cookies
    access_key = cookies.get("appAccessKey")
    client_name = cookies.get("clientName")
    
    # Fallback for dev
    DEV_ACCESS_KEY = os.getenv("DEV_ACCESS_KEY", "")
    CLIENT_NAME = os.getenv("CLIENT_NAME", "")
    
    if not access_key:
        access_key = DEV_ACCESS_KEY
    
    if not client_name:
        client_name = CLIENT_NAME

    source = "未知"
    if cookies.get("appAccessKey"):
        source = "来自用户 Cookie"
    elif DEV_ACCESS_KEY and access_key == DEV_ACCESS_KEY:
        source = "开发者本地调试 AK"
    
    if not access_key:
        return f"错误: 未找到 AccessKey。请确保通过 Bohrium 平台打开应用或配置了 DEV_ACCESS_KEY。\n来源: {source}"

    # bizNo 自动生成
    timestamp = int(time.time())
    rand_part = random.randint(1000, 9999)
    biz_no = int(f"{timestamp}{rand_part}")

    url = "https://openapi.dp.tech/openapi/v1/api/integral/consume"
    headers = {
        "accessKey": access_key,
        "x-app-key": client_name if client_name else "",
        "Content-Type": "application/json"
    }
    
    try:
        event_value = int(event_value)
        sku_id = int(sku_id)
    except ValueError:
        return "错误: 扣费数额和 SkuId 必须为整数"

    payload = {
        "bizNo": biz_no,
        "changeType": 1,
        "eventValue": event_value,
        "skuId": sku_id,
        "scene": "appCustomizeCharge"
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        result = resp.text
        # Try to format JSON
        try:
            res_json = resp.json()
            result = json.dumps(res_json, indent=2, ensure_ascii=False)
        except:
            pass
    except Exception as e:
        result = str(e)

    return f"AccessKey 来源: {source}\nAccessKey: {access_key[:6]}***\n\n接口返回:\n{result}"

APP_NAME = "agents"
SESSION_USER_ID = "web-user"
SESSION_ID = "default-session"

if USE_ADK:
    if InMemorySessionService is None:
        print("[Warn] InMemorySessionService 不可用，回退到简易 Runner")
        USE_ADK = False
    else:
        try:
            session_service = InMemorySessionService()
            runner = Runner(app_name=APP_NAME, agent=root_agent, session_service=session_service)
            _session_ready = False
            _session_lock = None  # 懒初始化锁
        except Exception as _runner_err:
            print(f"[Warn] 初始化 ADK Runner 失败，回退到简易 Runner: {_runner_err}")
            USE_ADK = False

if not USE_ADK:
    # 简易 Runner：直接调用底层模型（LiteLlm 或其它），包装为与 ADK 近似的事件结构
    class SimpleEventContentPart:
        def __init__(self, text: str):
            self.text = text

    class SimpleEventContent:
        def __init__(self, text: str):
            self.parts = [SimpleEventContentPart(text)]

    class SimpleEvent:
        def __init__(self, author: str, text: str):
            self.author = author
            self.content = SimpleEventContent(text)

    class SimpleRunner:
        def __init__(self, agent):
            self.agent = agent
        async def run_async(self, user_id: str, session_id: str, new_message, **kwargs):  # type: ignore
            try:
                parts = getattr(new_message, 'parts', [])
                user_text = "\n".join([getattr(p, 'text', '') for p in parts if getattr(p, 'text', '')]) or str(new_message)
            except Exception:
                user_text = str(new_message)
            model_obj = getattr(self.agent, 'model', None)
            reply = "[模型不可用]"
            if model_obj and hasattr(model_obj, 'generate_content'):
                try:
                    resp = model_obj.generate_content(user_text)
                    reply = getattr(resp, 'text', None) or (str(resp) if resp else "[空响应]")
                except Exception as e:
                    reply = f"[调用失败: {e}]"
            yield SimpleEvent(author=getattr(self.agent, 'name', 'agent'), text=reply)

    runner = SimpleRunner(root_agent)
    _session_ready = True  # 简易模式不做 session 管理
    _session_lock = None


async def _ensure_runner_session():
    if not USE_ADK:
        return
    global _session_ready, _session_lock
    if _session_ready:
        return
    if _session_lock is None:
        _session_lock_local = asyncio.Lock()
        if globals().get('_session_lock') is None:
            globals()['_session_lock'] = _session_lock_local
    async with _session_lock:  # type: ignore[arg-type]
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
    global _session_ready
    if not USE_ADK:
        _session_ready = True
        return
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


async def collect_agent_response(message_str: str, filter_mode: str = "strict", use_vision: bool = False, vision_model: str = "gpt-4o", use_pipeline: bool = False, time_window_days: Optional[int] = None, source_pref: str = "arxiv_leaderboard") -> list:
    """核心调用：增加超时与事件调试日志，避免长时间无明显反馈"""
    chunks = []
    start_ts = time.time()
    response_timeout_env = os.getenv("RESPONSE_TIMEOUT")
    try:
        # 默认等待时间由 45s 提升为 600s (10 分钟)，只在达到上限才判定超时
        # 可通过环境变量 RESPONSE_TIMEOUT 覆盖（单位：秒）
        timeout_sec = int(response_timeout_env) if response_timeout_env else 600
    except ValueError:
        timeout_sec = 600
    # 为下游 SDK 统一设置请求超时（litellm等），可由环境变量覆盖
    litellm_timeout_env = os.getenv("LITELLM_TIMEOUT")
    try:
        litellm_timeout = int(litellm_timeout_env) if litellm_timeout_env else 60
    except ValueError:
        litellm_timeout = 60
    os.environ["LITELLM_TIMEOUT"] = str(litellm_timeout)
    debug_events = os.getenv("ADK_DEBUG", "0").lower() in ("1", "true", "yes")
    try:
        await _ensure_runner_session()
        normalized_message = message_str if isinstance(message_str, str) else str(message_str)

        mode_hint = "\n[系统提示：当前过滤模式为" + ("严格模式" if filter_mode == "strict" else "宽松模式") + "。在调用 get_latest_sota 等工具时，请根据过滤模式决定是否放宽约束条件。宽松模式下，如果严格过滤没有结果，应自动放宽约束返回候选结果。]"

        vision_hint = ""
        if use_vision:
            vision_hint = f"\n[系统提示：已启用 Vision Model 增强提取（{vision_model}）。在调用 run_trustworthy_sota_search 时，请传递 use_vision=True 和 vision_model='{vision_model}' 参数以启用 Vision Model 处理复杂表格和图表。]"

        pipeline_hint = ""
        if use_pipeline and PIPELINE_AVAILABLE:
            pipeline_hint = "\n[系统提示：已启用 Multi-Agent Pipeline 模式。对于 SOTA 查询，请优先使用 run_trustworthy_sota_search 而不是 get_latest_sota。]"
        elif use_pipeline and not PIPELINE_AVAILABLE:
            pipeline_hint = "\n[系统提示：Pipeline 功能不可用，将回退到 get_latest_sota。]"

        recency_hint = ""
        if time_window_days:
            recency_hint = f"\n[系统提示：请在搜索阶段应用最近 {time_window_days} 天的时间窗，并按发布时间优先排序。]"

        source_hint = ""
        if source_pref:
            source_hint = f"\n[系统提示：来源偏好为 {source_pref}。请优先使用 {source_pref}，必要时再回退其它来源。]"

        enhanced_message = normalized_message.strip() + mode_hint + vision_hint + pipeline_hint + recency_hint + source_hint

        user_content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=enhanced_message)],
        )

        async def _run():
            async for event in runner.run_async(
                user_id=SESSION_USER_ID,
                session_id=SESSION_ID,
                new_message=user_content,
            ):
                chunks.append(event)
                if debug_events:
                    try:
                        print(f"[ADK-Event] 累计{len(chunks)}条 | 类型={type(event).__name__}")
                    except Exception:
                        pass
        try:
            await asyncio.wait_for(_run(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            print(f"[ADK-Timeout] 已等待 {timeout_sec}s 未完成，返回当前已收集分片 {len(chunks)}。可设置环境变量 RESPONSE_TIMEOUT 调整超时（秒），例如 1200 以等待 20 分钟。")
        elapsed = int(time.time() - start_ts)
        if debug_events:
            print(f"[ADK-Done] 总耗时 {elapsed}s, 分片 {len(chunks)}")

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"直接调用模型失败详情:\n{error_detail}")
        raise Exception(
            f"无法调用Agent。\n错误: {e}\n\n请检查Agent配置和API密钥是否正确。"
        )
    return chunks


async def chat_with_agent(message, history, filter_mode="严格模式", use_vision=False, vision_model="gpt-4o", use_pipeline=False, time_window_choice="不限", source_pref_choice="arXiv+Leaderboard", provider="Gemini", api_key: Optional[str] = None):
    if not message or not message.strip():
        return "", history

    history = history or []

    try:
        message_str = message if isinstance(message, str) else (str(message) if message else "")
        internal_mode = "relaxed" if filter_mode == "宽松模式" else "strict"

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

        def _format_sota_json_if_any(raw_text: str) -> str:
            """检测 SOTA JSON 并格式化为 Markdown 表格 + 链接。失败则原样返回。"""
            if not raw_text:
                return raw_text
            candidate = raw_text.strip()
            # 仅在看起来像 JSON 时尝试
            if not (candidate.startswith('{') and candidate.endswith('}')):
                return raw_text
            import json as _json
            try:
                data = _json.loads(candidate)
            except Exception:
                return raw_text
            # -------- Pipeline 结果格式化 (Multi-Agent) --------
            # 特征：包含 keys: status==success, summary(dict), papers(list)
            if isinstance(data, dict) and data.get('status') == 'success' and isinstance(data.get('papers'), list) and 'summary' in data and 'sota' not in data:
                summary = data.get('summary', {})
                papers = data.get('papers', [])
                conflicts = data.get('conflicts', [])
                q = data.get('query') or data.get('benchmark') or '未命名查询'
                lines = []
                lines.append(f"### 🔄 可信 Pipeline 汇总：{q}")
                lines.append(f"**处理论文数**：{summary.get('total_papers_processed','?')}  ｜ **提取指标总数**：{summary.get('total_metrics_extracted','?')}  ｜ **发现冲突**：{summary.get('conflicts_found','0')}")
                if not papers:
                    lines.append('\n_未找到可格式化的论文结果_')
                    return '\n'.join(lines)
                # 表头：序号 / 标题 / 指标(前3) / 主指标值 / arXiv
                lines.append('\n| # | 标题 | 指标(前3) | 主指标(猜测) | arXiv |')
                lines.append('|---|------|-----------|-------------|-------|')
                for idx, p in enumerate(papers, 1):
                    if not isinstance(p, dict):
                        continue
                    title = (p.get('title') or '无标题').replace('|', ' ')[:120]
                    pid = p.get('paper_id') or ''
                    # 构造 arXiv 链接（若 short id 符合 pattern）
                    arxiv_link = '—'
                    if pid and len(pid) >= 5 and pid[0].isdigit():
                        arxiv_link = f"[链接](https://arxiv.org/abs/{pid})"
                    metrics = p.get('metrics') or []
                    metric_names: list[str] = []
                    for _m in metrics[:3]:
                        if isinstance(_m, dict):
                            mv = _m.get('metric')
                            if isinstance(mv, str):
                                metric_names.append(mv)
                    metrics_cell = ', '.join(metric_names) if metric_names else '—'
                    # 猜测主指标：ao/sr/auc/map/accuracy/f1_score/top1_accuracy 按优先级
                    primary_val = '—'
                    preferred_order = ['ao','sr','auc','map','accuracy','f1_score','top1_accuracy']
                    metric_map = {}
                    for m in metrics:
                        if isinstance(m, dict):
                            metric_map[m.get('metric')] = m.get('value')
                    for k in preferred_order:
                        v = metric_map.get(k)
                        if isinstance(v, (int,float)):
                            primary_val = f"{v:.2f}%" if v > 1 else f"{v*100:.2f}%"
                            break
                    lines.append(f"| {idx} | {title} | {metrics_cell} | {primary_val} | {arxiv_link} |")
                # 冲突汇总
                if isinstance(conflicts, list) and conflicts:
                    lines.append('\n#### ⚠️ 冲突概览 (Top 5)')
                    lines.append('| 指标 | 差异 | 等级 | 涉及论文数 |')
                    lines.append('|-------|------|------|-----------|')
                    for cf in conflicts[:5]:
                        if not isinstance(cf, dict):
                            continue
                        lines.append(f"| {cf.get('metric','?')} | {cf.get('difference','?')} | {cf.get('conflict_level','?')} | {cf.get('papers_involved','?')} |")
                return '\n'.join(lines).strip()
            # 判定是 SOTA 结构
            if not isinstance(data, dict) or 'sota' not in data or not isinstance(data.get('sota'), dict):
                return raw_text
            sota = data.get('sota') or {}
            top = data.get('top_candidates') or []
            benchmark = data.get('benchmark') or data.get('query') or '未知基准'
            lines = []
            lines.append(f"### 📌 {benchmark} 最新 SOTA")
            # SOTA 主行
            sid = sota.get('id') or 'N/A'
            title = sota.get('title') or '无标题'
            arxiv_url = sota.get('arxiv_url') or (f"https://arxiv.org/abs/{sid}" if sid and sid != 'N/A' else '')
            pdf_url = sota.get('pdf_url') or ''
            metric = sota.get('metric')
            metric_str = f"{metric:.2f}" if isinstance(metric, (int, float)) else (str(metric) if metric is not None else '—')
            lines.append("**SOTA 模型**：" + (f"[{title}]({arxiv_url})" if arxiv_url else title))
            if pdf_url and pdf_url != arxiv_url:
                lines.append(f"**PDF**：[{pdf_url}]({pdf_url})")
            if metric_str:
                lines.append(f"**主指标**：{metric_str}")
            datasets = sota.get('datasets') or []
            if datasets:
                lines.append("**数据集**：" + ", ".join(datasets))
            scopes = sota.get('scopes') or []
            if scopes:
                lines.append("**范式/范围**：" + ", ".join(scopes))
            lines.append("")
            # 候选表格
            if isinstance(top, list) and top:
                lines.append("#### 🔎 Top 候选 (最多 5 条)")
                lines.append("| # | 标题 | 指标 | 数据集 | arXiv | PDF |")
                lines.append("|---|-------|------|--------|-------|-----|")
                for idx, c in enumerate(top, 1):
                    if not isinstance(c, dict):
                        continue
                    cid = c.get('id') or ''
                    ctitle = (c.get('title') or '').replace('|', ' ')[:120]
                    cmetric = c.get('metric')
                    cmetric_str = f"{cmetric:.2f}" if isinstance(cmetric, (int, float)) else (str(cmetric) if cmetric is not None else '—')
                    cdsets = c.get('datasets') or []
                    cdsets_str = ",".join(cdsets) if cdsets else '—'
                    carxiv = c.get('arxiv_url') or (f"https://arxiv.org/abs/{cid}" if cid else '')
                    carxiv_link = f"[链接]({carxiv})" if carxiv else '—'
                    cpdf = c.get('pdf_url') or ''
                    cpdf_link = f"[PDF]({cpdf})" if cpdf else '—'
                    lines.append(f"| {idx} | {ctitle} | {cmetric_str} | {cdsets_str} | {carxiv_link} | {cpdf_link} |")
            # 冲突信息（可选）
            verification = data.get('verification') or {}
            conflicts = verification.get('conflicts') or []
            if conflicts:
                lines.append("")
                lines.append("#### ⚠️ 指标冲突摘要")
                lines.append("| 指标 | 差异 | 等级 | 涉及论文数 |")
                lines.append("|-------|------|------|-----------|")
                for cf in conflicts[:5]:
                    if not isinstance(cf, dict):
                        continue
                    lines.append(f"| {cf.get('metric','?')} | {cf.get('difference','?')} | {cf.get('conflict_level','?')} | {cf.get('papers_involved','?')} |")
            formatted = "\n".join(lines).strip()
            return formatted if formatted else raw_text

        

        # 根据前端输入设置对应的环境变量（仅当前进程生效）
        provider_norm = (provider or "GPT").strip().lower()
        provided_key = (api_key or "").strip()
        if not provided_key:
            history.append((message, "❌ 未提供 API Key。请在右侧输入框填写后再试。"))
            return "", history
        # 清理可能遗留的环境变量，避免串号
        for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
            if os.getenv(k):
                os.environ.pop(k, None)
        if provider_norm == "gpt":
            os.environ["OPENAI_API_KEY"] = provided_key
        elif provider_norm == "deepseek":
            os.environ["DEEPSEEK_API_KEY"] = provided_key
        elif provider_norm == "qwen":
            os.environ["DASHSCOPE_API_KEY"] = provided_key
        elif provider_norm == "gemini":
            os.environ["GEMINI_API_KEY"] = provided_key
        else:
            history.append((message, f"❌ 未知提供商: {provider}."))
            return "", history

        # 设置统一的提供商环境标识，供 agent.py 或后续切换使用
        os.environ["LLM_PROVIDER"] = provider_norm

        # 动态切换 root_agent 使用的模型，避免首次导入时锁死
        if LiteLlm is not None:
            try:
                current_model_name = getattr(getattr(root_agent, "model", None), "model", "")
                target_model_name = None
                if provider_norm == "gpt":
                    target_model_name = "openai/gpt-4o-mini"
                elif provider_norm == "deepseek":
                    target_model_name = "deepseek/deepseek-chat"
                elif provider_norm == "qwen":
                    target_model_name = "qwen/qwen-plus"
                elif provider_norm == "gemini":
                    target_model_name = "gemini/gemini-2.5-flash"
                if target_model_name and target_model_name != current_model_name:
                    root_agent.model = LiteLlm(model=target_model_name)
                    print(f"[Model-Switch] 模型已切换为 {target_model_name}")
            except Exception as switch_err:
                print(f"[Model-Switch] 切换模型失败: {switch_err}")
        else:
            print("[Model-Switch] LiteLlm 不可用，无法动态切换模型。")

        # 调用 Agent（保持在当前事件循环中，避免跨线程/跨事件循环）
        _start_ts_local = time.time()
        chunks = await collect_agent_response(
            message_str,
            internal_mode,
            use_vision,
            vision_model,
            use_pipeline,
            None if time_window_choice == "不限" else (180 if time_window_choice == "180 天" else 365),
            "arxiv_leaderboard" if source_pref_choice == "arXiv+Leaderboard" else ("scholar" if source_pref_choice == "Scholar" else "arxiv")
        )

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
                elif not isinstance(chunk, str) and hasattr(chunk, 'content'):
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
            response = "⚠️ Agent可能仍在处理或未返回可解析内容。可稍后重试，或设置环境变量 RESPONSE_TIMEOUT 调整等待秒数。"

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
        response = _format_sota_json_if_any(response)

        if not response or not response.strip():
            response = "抱歉，我没有理解您的问题。请尝试重新表述您的问题。"

        # 追加响应耗时提示（可选）
        try:
            latency = int(time.time() - _start_ts_local)
            response += f"\n\n⏱️ 响应耗时约 {latency}s"
        except Exception:
            pass

        history.append((message, response))
        return "", history
    except Exception as e:
        error_msg = f"❌ 处理请求时出错: {str(e)}\n\n请检查：\n1. API密钥是否正确配置\n2. 网络连接是否正常\n3. 输入的问题是否有效"
        history.append((message, error_msg))
        return "", history


async def clear_chat():
    await _reset_runner_session()
    return [], ""


# 单一浅色主题 + 全宽布局（不使用 @import）
custom_css = """
:root {
  color-scheme: light;
  --bg: #f7f9fc;
  --panel: #ffffff;
  --card: #ffffff;
  --border: #e5e7eb;
  --text: #0f172a;
  --muted: #64748b;
  --accent: #2563eb;
  --accent-2: #3b82f6;
  --ring: rgba(37, 99, 235, 0.2);
}

* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: system-ui, -apple-system, 'Segoe UI', Arial, sans-serif; line-height: 1.6; }

.gradio-container { max-width: 100% !important; width: 100% !important; margin: 0 !important; padding: 20px 24px 36px !important; background: var(--bg) !important; }
.gradio-row, .gradio-column, .gradio-block, .tabitem, .tabs, .tab-nav, .prose, .block, .form, .container { background: transparent !important; border-color: var(--border) !important; }

.hero { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 20px; }
.hero .eyebrow { color: var(--muted); font-size: 12px; letter-spacing: .12em; text-transform: uppercase; }
.hero h1 { margin: 6px 0 8px; font-size: 26px; font-weight: 700; }
.hero p { color: var(--muted); margin: 0; }

.stat-grid { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin: 14px 0 8px; }
.stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px; text-align: center; }
.stat-value { font-size: 22px; font-weight: 700; color: var(--text); }
.stat-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }

.gradio-row { gap: 16px !important; }

.gradio-chatbot { background: var(--panel) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; padding: 10px !important; }
.gradio-chatbot .user { background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important; color: #ffffff !important; border: none !important; }
.gradio-chatbot .bot { background: var(--card) !important; border: 1px solid var(--border) !important; }

.gradio-textbox textarea { background: var(--panel) !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: 10px !important; padding: 12px 14px !important; outline: none !important; box-shadow: none !important; }
.gradio-textbox textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--ring) !important; }

.gradio-button { border-radius: 10px !important; font-weight: 600 !important; }
.gradio-button.primary { background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important; color: #ffffff !important; border: none !important; }
.gradio-button.secondary { background: transparent !important; color: var(--text) !important; border: 1px solid var(--border) !important; }

.gradio-accordion { background: var(--panel) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
.gradio-accordion .gradio-accordion-header { color: var(--text) !important; }

.sidebar-card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
.sidebar-card h3 { margin: 0 0 8px; font-size: 16px; }
.prompt-list li { color: var(--muted); border-bottom: 1px dashed var(--border); }
.prompt-list li:last-child { border-bottom: none; }

.footer { color: var(--muted) !important; border-top: 1px solid var(--border); }

@media (max-width: 860px) { .stat-grid { grid-template-columns: 1fr; } }
"""


# 创建Gradio界面（无主题切换，仅浅色）
with gr.Blocks(
    title="SotaAgent - SOTA模型查询助手",
    theme=gr.themes.Soft(),
    css=custom_css,
) as iface:

    gr.Markdown(
        """
        <div class=\"hero\">\n            <p class=\"eyebrow\">SotaAgent · 研究辅助面板</p>\n            <h1>精准检索基准 · 秒回最新 SOTA · 中文交互更自然</h1>\n            <p>整合 arXiv、Benchmark 配置与自定义工具链，帮助你快速定位实验表格、指标与模型亮点，支持自然语言与参数化双模式。</p>\n        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=8):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            # 新的统一资源查找顺序：assets/avatar.png -> 当前目录 png -> 父目录原始长文件名 -> emoji
            candidate_paths = [
                os.path.join(current_dir, "assets", "avatar.png"),
                os.path.join(current_dir, "avatar.png"),
                os.path.join(parent_dir, "人工智能_ 人工智能_ 自动机_ 脑_ 数码产品_ 机器人学_爱给网_aigei_com.png"),
                os.path.join(current_dir, "人工智能_ 人工智能_ 自动机_ 脑_ 数码产品_ 机器人学_爱给网_aigei_com.png"),
            ]
            avatar_image_path = "🤖"
            for pth in candidate_paths:
                if os.path.exists(pth):
                    avatar_image_path = pth
                    break

            chatbot = gr.Chatbot(label="", height=600, avatar_images=(None, avatar_image_path), show_copy_button=True, container=True)

            with gr.Row():
                msg = gr.Textbox(label="", placeholder="输入您的问题，例如：找 GOT-10k 上最近的纯监督 SOTA", scale=9, lines=3)
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
            provider_radio = gr.Radio(
                choices=["GPT", "DeepSeek", "Qwen", "Gemini"],
                value="GPT",
                label="🔑 模型提供商",
                info="选择你要使用的大模型提供商",
            )
            api_key_box = gr.Textbox(
                label="API Key",
                placeholder="在此粘贴你的 API 密钥（仅本次会话使用）",
                type="password",
            )
            filter_mode_radio = gr.Radio(
                choices=["严格模式", "宽松模式"],
                value="严格模式",
                label="🔍 过滤模式",
                info="严格模式：精确匹配所有约束条件；宽松模式：如果严格过滤无结果，自动放宽约束返回候选",
            )

            pipeline_available_display = "✅ 可用" if PIPELINE_AVAILABLE else "❌ 不可用（需要安装依赖）"
            with gr.Accordion(f"🔄 Multi-Agent Pipeline（可选）{pipeline_available_display}", open=False):
                use_pipeline_checkbox = gr.Checkbox(
                    value=False,
                    label="启用 Multi-Agent Pipeline",
                    info="使用多智能体协作流程进行更可靠的 SOTA 验证（Scanner → Extractor → Normalizer → Verifier）",
                    interactive=PIPELINE_AVAILABLE,
                )
                if not PIPELINE_AVAILABLE:
                    gr.Markdown("<div style='font-size: 0.85em; color: #b45309;'>⚠️ 运行 Pipeline 前请安装：<code>pip install -r My_First_Agent/requirements_pipeline.txt</code></div>")

            # 时间窗与来源偏好控件
            with gr.Accordion("⏱️ 时间窗与来源偏好", open=False):
                time_window_radio = gr.Radio(
                    choices=["不限", "180 天", "365 天"],
                    value="不限",
                    label="时间窗"
                )
                source_pref_radio = gr.Radio(
                    choices=["arXiv+Leaderboard", "arXiv", "Scholar"],
                    value="arXiv+Leaderboard",
                    label="来源偏好"
                )

            with gr.Accordion("🤖 Vision Model 增强（Beta）", open=False):
                use_vision_checkbox = gr.Checkbox(value=False, label="启用 Vision Model", info="处理复杂表格和图表（成本较高）")
                vision_model_radio = gr.Radio(choices=["gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash-exp"], value="gpt-4o", label="Vision Model 选择")

            with gr.Accordion("💰 光子支付测试", open=False):
                gr.Markdown("测试光子扣费接口。请确保已获取 AccessKey (通过 Bohrium 打开)。")
                pay_amount = gr.Number(label="扣费数额 (eventValue)", value=0, precision=0)
                pay_sku = gr.Number(label="SkuId", value=0, precision=0)
                pay_btn = gr.Button("提交扣费请求")
                pay_result = gr.Textbox(label="接口返回", lines=5)
                
                pay_btn.click(
                    fn=charge_photon,
                    inputs=[pay_amount, pay_sku],
                    outputs=[pay_result]
                )

            gr.Markdown(
                """
                <div class=\"sidebar-card\">\n                    <h3>🎯 高效提问技巧</h3>\n                    <ul class=\"prompt-list\">\n                        <li>描述 Benchmark + 时间窗口：例如 “GOT-10k 最近 180 天 SOTA”。</li>\n                        <li>加上约束：纯监督 / 零样本 / 不含额外数据。</li>\n                        <li>询问论文时附上 arXiv ID（如 2305.00012）。</li>\n                        <li>需要表格输出时附加 “请整理成表格”。</li>\n                    </ul>\n                </div>
                """
            )

    # 交互事件
    msg.submit(
        fn=chat_with_agent,
        inputs=[msg, chatbot, filter_mode_radio, use_vision_checkbox, vision_model_radio, use_pipeline_checkbox, time_window_radio, source_pref_radio, provider_radio, api_key_box],
        outputs=[msg, chatbot],
        api_name=False
    )
    submit_btn.click(
        fn=chat_with_agent,
        inputs=[msg, chatbot, filter_mode_radio, use_vision_checkbox, vision_model_radio, use_pipeline_checkbox, time_window_radio, source_pref_radio, provider_radio, api_key_box],
        outputs=[msg, chatbot],
        api_name=False
    )
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg],
        api_name=False
    )

    gr.Markdown("""
    <div class=\"footer\">\n      <p>💡 使用自然语言提问，Agent 会自动调用工具。</p>\n      <p>⚠️ 查询 SOTA 可能较慢，如遇速率限制请稍候重试。</p>\n    </div>
    """)


if __name__ == "__main__":
    # 在玻尔 Bohrium 平台部署时，需要固定使用 0.0.0.0:50001 端口对外提供服务
    # 这里默认使用 50001 端口，如需本地调试其它端口，可通过环境变量 BOHRIUM_PORT 覆盖
    port_env = os.getenv("BOHRIUM_PORT")
    try:
        port = int(port_env) if port_env else 50001
    except ValueError:
        print(f"环境变量 BOHRIUM_PORT 非法，回退到默认端口 50001，当前值: {port_env}")
        port = 50001

    # 获取 share 参数
    share_env = os.getenv("GRADIO_SHARE")
    share = True if share_env and share_env.lower() in ('true', '1', 'yes') else False

    print(f"正在启动 Gradio 服务... (Share={share}, Port={port})")
    if share:
        print("注意：开启 Share 模式可能会导致启动缓慢，因为需要下载 FRPC 二进制文件并建立隧道。如果长时间卡住，请尝试关闭 Share 模式。")

    iface.launch(
        server_name='0.0.0.0',
        server_port=port,
        share=share,
        show_error=True,
        favicon_path=None,
        show_api=False,
    )

