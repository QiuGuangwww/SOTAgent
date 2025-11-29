from google.adk.agents import Agent

import arxiv
import json
import os
from typing import List, Dict, Any, Tuple, Set, Optional
import re
import time
from datetime import datetime, timedelta
from google.adk.models.lite_llm import LiteLlm

# 尝试导入网络搜索相关库
try:
    from googlesearch import search as google_search
    GOOGLE_SEARCH_AVAILABLE = True
except ImportError as e:
    GOOGLE_SEARCH_AVAILABLE = False
    print(f"[Info] googlesearch-python 未安装或导入失败: {e}")
    print("[Info] 网络检索功能将使用备用方案。安装: pip install googlesearch-python")
except Exception as e:
    GOOGLE_SEARCH_AVAILABLE = False
    print(f"[Warning] googlesearch-python 导入时出错: {e}")
    print("[Info] 网络检索功能将使用备用方案。")

try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError as e:
    REQUESTS_AVAILABLE = False
    print(f"[Info] requests/beautifulsoup4 未安装或导入失败: {e}")
    print("[Info] 网络检索功能将受限。安装: pip install requests beautifulsoup4")
except Exception as e:
    REQUESTS_AVAILABLE = False
    print(f"[Warning] requests/beautifulsoup4 导入时出错: {e}")

# 导入 Pipeline 工具（可选，如果可用）
try:
    from .pipeline_tools import run_trustworthy_sota_search
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    print(f"[Warning] Multi-Agent Pipeline 不可用: {e}")
    print("[Warning] 如需启用，请安装依赖: pip install -r My_First_Agent/requirements_pipeline.txt")

PAPER_DIR="papers"
CACHE_DIR = os.path.join(PAPER_DIR, "web_search_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 网络搜索缓存（避免重复查询）
_web_search_cache: Dict[str, Dict[str, Any]] = {}
_cache_file = os.path.join(CACHE_DIR, "name_type_cache.json")

def _load_search_cache() -> Dict[str, Dict[str, Any]]:
    """加载网络搜索缓存"""
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"]
    for enc in encodings:
        try:
            with open(_cache_file, "r", encoding=enc, errors=("strict" if enc not in ("latin-1",) else "ignore")) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        except Exception:
            continue
    return {}

def _save_search_cache() -> None:
    """保存网络搜索缓存"""
    try:
        with open(_cache_file, "w", encoding="utf-8") as f:
            json.dump(_web_search_cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Warning] 保存搜索缓存失败: {e}")

# 加载缓存
_web_search_cache = _load_search_cache()

def _fetch_arxiv_results(client: arxiv.Client, search: arxiv.Search, max_retries: int = 3, base_delay: float = 3.0) -> List[arxiv.Result]:
    """
    安全获取 arXiv 查询结果，自动处理429等限流错误。
    """
    # 避免代理干扰 arXiv 访问
    try:
        import os
        for k in ("NO_PROXY", "no_proxy"):
            hosts = os.environ.get(k, "")
            hostset = {h.strip() for h in hosts.split(",") if h.strip()}
            hostset.update({"export.arxiv.org", "arxiv.org"})
            os.environ[k] = ",".join(sorted(hostset))
    except Exception:
        pass

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return list(client.results(search))
        except arxiv.HTTPError as err:
            last_error = err
            status = getattr(err, "status", None)
            if status in (429, 500, 502, 503):
                wait = base_delay * (attempt + 1)
                print(f"[arXiv] HTTP {status}，等待 {wait:.1f}s 后重试 (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            raise
        except Exception as err:
            # 处理代理相关错误或远端关闭
            last_error = err
            msg = str(err).lower()
            if any(tok in msg for tok in ["proxy", "remote end closed", "connection aborted"]):
                wait = base_delay * (attempt + 1)
                print(f"[arXiv] 连接/代理异常，等待 {wait:.1f}s 后重试 (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("arXiv 接口请求过于频繁，请稍后重试或缩小搜索范围。") from last_error

def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        List of paper IDs found in the search
    """
    
    # Use arxiv to find the papers 
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    try:
        papers = _fetch_arxiv_results(client, search)
    except RuntimeError as err:
        print(f"[search_papers] {err}")
        return []
    
    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    papers_info = {}
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc, errors=("strict" if enc not in ("latin-1",) else "ignore")) as json_file:
                papers_info = json.load(json_file)
            break
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            papers_info = {}
            continue
        except Exception:
            continue

    # Process each paper and add to papers_info  
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info
    
    # Save updated papers_info to json file
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(papers_info, json_file, indent=2, ensure_ascii=False)
    
    print(f"Results are saved in: {file_path}")
    
    return paper_ids

def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.
    
    Args:
        paper_id: The ID of the paper to look for
        
    Returns:
        JSON string with paper information if found, error message if not found
    """
 
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"]
                loaded = None
                for enc in encodings:
                    try:
                        with open(file_path, "r", encoding=enc, errors=("strict" if enc not in ("latin-1",) else "ignore")) as json_file:
                            papers_info = json.load(json_file)
                            if paper_id in papers_info:
                                loaded = papers_info[paper_id]
                                break
                    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    except Exception:
                        continue
                if loaded is not None:
                    return json.dumps(loaded, indent=2, ensure_ascii=False)
    
    return f"There's no saved information related to paper {paper_id}."

def _ensure_topic_dir(topic: str) -> Tuple[str, str]:
    safe = topic.lower().replace(" ", "_")
    path = os.path.join(PAPER_DIR, safe)
    os.makedirs(path, exist_ok=True)
    return path, os.path.join(path, "papers_info.json")

def _load_json(path: str) -> Dict[str, Any]:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors=("strict" if enc not in ("latin-1",) else "ignore")) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        except Exception:
            continue
    return {}

def _save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def _paper_to_dict(p: arxiv.Result) -> Dict[str, Any]:
    abs_url = None
    try:
        # arXiv 标准摘要链接
        abs_url = f"https://arxiv.org/abs/{p.get_short_id()}"
    except Exception:
        abs_url = None
    return {
        "id": p.get_short_id(),
        "title": p.title,
        "authors": [a.name for a in p.authors],
        "summary": p.summary,
        "pdf_url": p.pdf_url,
        "arxiv_url": abs_url,
        "primary_category": getattr(p, "primary_category", None),
        "categories": list(getattr(p, "categories", []) or []),
        "published": str(p.published.date()) if p.published else None,
        "updated": p.updated.isoformat() if getattr(p, "updated", None) else None,
    }

def find_papers_by_benchmark(benchmark: str, max_results: int = 50) -> List[str]:
    """
    按基准/细分领域关键词检索论文，保存信息，并返回按时间倒序的短 ID 列表。
    """
    client = arxiv.Client()
    # 提高命中率：在标题与摘要匹配
    query = f'ti:"{benchmark}" OR abs:"{benchmark}"'
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    try:
        results = _fetch_arxiv_results(client, search)
    except RuntimeError as err:
        print(f"[find_papers_by_benchmark] {err}")
        return []
    # 最新在前
    results.sort(key=lambda r: r.published or datetime.min, reverse=True)

    path, file_path = _ensure_topic_dir(benchmark)
    store = _load_json(file_path)

    paper_ids: List[str] = []
    for p in results:
        d = _paper_to_dict(p)
        store[d["id"]] = d
        paper_ids.append(d["id"])

    _save_json(file_path, store)
    print(f"Saved {len(paper_ids)} papers to: {file_path}")
    return paper_ids

_METRIC_PATTERNS = [
    r"(?:accuracy|acc)\s*(?:of|=|:)?\s*(\d{1,2}(?:\.\d+)?|\d{3})(?:\s*%|\b)",
    r"(?:f1|f1-score)\s*(?:of|=|:)?\s*(\d{1,2}(?:\.\d+)?|\d{3})(?:\s*%|\b)",
    r"(?:bleu)\s*(?:of|=|:)?\s*(\d{1,2}(?:\.\d+)?)(?:\s*%|\b)",
    r"(?:rouge[-_]?(?:l|1|2)?)\s*(?:of|=|:)?\s*(\d{1,2}(?:\.\d+)?)(?:\s*%|\b)",
    r"(?:mmlu|mmmu|mmlu_score)\s*(?:of|=|:)?\s*(\d{1,2}(?:\.\d+)?|\d{3})(?:\s*%|\b)",
]
_SOTA_HINTS = ["state-of-the-art", "sota", "sets new state of the art", "new sota", "achiev", "surpass", "outperform"]

def _extract_metric(text: str) -> Tuple[str, float]:
    text_l = text.lower()
    for pat in _METRIC_PATTERNS:
        m = re.search(pat, text_l)
        if m:
            val = m.group(1)
            try:
                score = float(val)
                # 百分数统一到 0-100
                if score <= 1.0:
                    score *= 100.0
                return pat, score
            except ValueError:
                continue
    return "", -1.0

def _looks_like_sota(title: str, summary: str) -> bool:
    blob = (title + " " + summary).lower()
    return any(h in blob for h in _SOTA_HINTS)

_SCOPE_PATTERNS: Dict[str, List[str]] = {
    "self-supervised": [
        r"\bself[- ]?supervised\b", r"\bunsupervised pretraining\b", r"\bssl\b"
    ],
    "supervised": [
        r"\bsupervised\b", r"\bfully[- ]?supervised\b"
    ],
    "reinforcement": [
        r"\breinforcement learning\b", r"\brl\b"
    ],
    "semi-supervised": [
        r"\bsemi[- ]?supervised\b"
    ],
    "weakly-supervised": [
        r"\bweakly[- ]?supervised\b", r"\bweak supervision\b"
    ],
    "unsupervised": [
        r"\bunsupervised\b"
    ],
    "zero-shot": [
        r"\bzero[- ]?shot\b"
    ],
    "few-shot": [
        r"\bfew[- ]?shot\b", r"\bone[- ]?shot\b"
    ],
}

def _detect_scopes(title: str, summary: str) -> Set[str]:
    text = f"{title}\n{summary}".lower()
    found: Set[str] = set()
    for scope, pats in _SCOPE_PATTERNS.items():
        for pat in pats:
            if re.search(pat, text):
                found.add(scope)
                break
    return found

# ------------------- 通用 Constraints 解析（范式/数据/模态/tricks/资源） -------------------
_DATA_REGIME_PATTERNS: Dict[str, List[str]] = {
    "no-extra-data": [r"\bno extra data\b", r"\bwithout extra data\b", r"\btraining data\b.*\bonly\b", r"\bno external\b"],
    "extra-data": [r"\bextra data\b", r"\bexternal data\b", r"\badditional data\b", r"\bweb[- ]?scale\b"],
    "pretrained": [r"\bpre[- ]?train", r"\bpretrained\b", r"\bfoundation model\b"],
    "distilled": [r"\bdistill", r"\bteacher[- ]?student\b"],
}
_MODALITY_PATTERNS: Dict[str, List[str]] = {
    "rgb": [r"\brgb\b"],
    "rgbd": [r"\brgb[- ]?d\b"],
    "multimodal": [r"\bmultimodal\b", r"\bvision[- ]?language\b"],
    "infrared": [r"\bir\b", r"\binfrared\b", r"\bthermal\b"],
    "event": [r"\bevent camera\b", r"\bevent[- ]?based\b"],
}
_TRICKS_PATTERNS: Dict[str, List[str]] = {
    "tta": [r"\btest[- ]?time augmentation\b", r"\btta\b"],
    "ensemble": [r"\bensemble\b"],
    "prompting": [r"\bprompt[- ]?(tuning|engineering)\b"],
}
_RESOURCE_PATTERNS: Dict[str, List[str]] = {
    "realtime": [r"\breal[- ]?time\b", r"\b\d+\s*fps\b"],
    "lightweight": [r"\b(lightweight|tiny|small) model\b", r"\b<\s*\d+\s*(m|b)\s*params\b"],
}

def _extract_constraints(title: str, summary: str) -> Dict[str, Any]:
    text = f"{title}\n{summary}".lower()
    def match_dict(pats: Dict[str, List[str]]) -> List[str]:
        out: List[str] = []
        for k, lst in pats.items():
            if any(re.search(p, text) for p in lst):
                out.append(k)
        return sorted(list(set(out)))
    return {
        "scopes": sorted(list(_detect_scopes(title, summary))),
        "data_regime": match_dict(_DATA_REGIME_PATTERNS),
        "modality": match_dict(_MODALITY_PATTERNS),
        "tricks": match_dict(_TRICKS_PATTERNS),
        "resources": match_dict(_RESOURCE_PATTERNS),
    }

# ------------------- 数据集与指标抽取及方向配置 -------------------
_DATASET_PATTERNS: Dict[str, List[str]] = {
    # 跟踪常见数据集
    "LaSOT": [r"\blasot\b"],
    "GOT-10k": [r"\bgot[- ]?10k\b"],
    "OTB": [r"\botb(?:50|100)?\b"],
    "TrackingNet": [r"\btrackingnet\b"],
    "TNL2K": [r"\btnl2k\b"],
    # VLA (Vision-Language-Action) 常见数据集
    # 注意：RT-1, RT-2, RT-X 是模型名，不是数据集，已从数据集中移除
    "Open-X Embodiment": [r"\bopen[- ]?x embodiment\b", r"\bopenx\b"],
    "Bridge": [r"\bbridge dataset\b", r"\bbridge v2\b"],
    "LIBERO": [r"\blibero\b"],
    "Calvin": [r"\bcalvin\b"],
    "Language-Table": [r"\blanguage[- ]?table\b", r"\blangtable\b"],
    "ALFRED": [r"\balfred\b"],
    "Meta-World": [r"\bmeta[- ]?world\b"],
    # 注意：SayCan, BC-Z, PaLM-E, Gato 是模型名，不是数据集，已从数据集中移除
    "MOO": [r"\bmoo\b", r"\bmultimodal open[- ]?world\b"],
    "RoboMimic": [r"\brobomimic\b"],
    "MIME": [r"\bmime\b"],
    "RoboTurk": [r"\broboturk\b"],
    "Dactyl": [r"\bdactyl\b"],
    "RLBench": [r"\brlbench\b"],
    "WidowX": [r"\bwidowx\b"],
    "PALM-E": [r"\bpalm[- ]?e\b"],
}
_DATASET_METRIC_CONFIG: Dict[str, Dict[str, Any]] = {
    # 每个数据集的主指标与方向（True=越大越好）
    "LaSOT": {"primary": ["success", "precision", "success_rate"], "larger_is_better": True},
    "GOT-10k": {"primary": ["ao", "success", "mAP"], "larger_is_better": True},
    "OTB": {"primary": ["success", "precision"], "larger_is_better": True},
    "TrackingNet": {"primary": ["success", "precision"], "larger_is_better": True},
    "TNL2K": {"primary": ["success", "precision"], "larger_is_better": True},
    # VLA 数据集指标配置
    # 注意：RT-1, RT-2, RT-X 是模型名，不是数据集，已从指标配置中移除
    "Open-X Embodiment": {"primary": ["success_rate", "task_success"], "larger_is_better": True},
    "Bridge": {"primary": ["success_rate", "task_success"], "larger_is_better": True},
    "LIBERO": {"primary": ["success_rate", "task_success"], "larger_is_better": True},
    "Calvin": {"primary": ["success_rate", "task_success"], "larger_is_better": True},
    "Language-Table": {"primary": ["success_rate", "task_success"], "larger_is_better": True},
    "ALFRED": {"primary": ["success_rate", "goal_condition_success"], "larger_is_better": True},
    "Meta-World": {"primary": ["success_rate", "task_success"], "larger_is_better": True},
    "RLBench": {"primary": ["success_rate", "task_success"], "larger_is_better": True},
}

# VLA 常见数据集列表（真实的数据集）
_VLA_BENCHMARKS = [
    "Open-X Embodiment", "Bridge", 
    "LIBERO", "Calvin", "Language-Table", "ALFRED", "Meta-World",
    "MOO", "RoboMimic", "MIME", "RoboTurk", "Dactyl", "RLBench", "WidowX"
]

# 查询模式：用于判断一个名称是数据集还是模型
# 数据集模式：这些模式表明后面的名称是数据集
_DATASET_INDICATOR_PATTERNS = [
    r"(?:在|on|evaluated\s+on|tested\s+on|benchmark|数据集|dataset)\s+([A-Za-z0-9\-\+_\/ ]{1,32})",
    r"([A-Za-z0-9\-\+_\/ ]{1,32})\s+(?:上的|上|on|dataset|数据集|benchmark|基准)",
    r"(?:sota|state[- ]of[- ]the[- ]art|performance|accuracy|score)\s+(?:on|在)\s+([A-Za-z0-9\-\+_\/ ]{1,32})",
    r"([A-Za-z0-9\-\+_\/ ]{1,32})\s+(?:数据集|dataset|benchmark|基准)",
]

# 模型模式：这些模式表明后面的名称是模型
_MODEL_INDICATOR_PATTERNS = [
    r"([A-Za-z0-9\-\+_\/ ]{1,32})\s+(?:模型|model|architecture|架构|proposes|introduces|presents)",
    r"(?:proposes|introduces|presents|develops|designs)\s+([A-Za-z0-9\-\+_\/ ]{1,32})",
    r"([A-Za-z0-9\-\+_\/ ]{1,32})\s+(?:achieves|obtains|reaches|gets)",
]

def _fetch_search_result_content(url: str, timeout: int = 5) -> str:
    """获取搜索结果的页面内容（标题和摘要）"""
    if not REQUESTS_AVAILABLE:
        return ""
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            # 提取标题
            title = soup.find("title")
            title_text = title.get_text().strip() if title else ""
            # 提取 meta description
            meta_desc = soup.find("meta", attrs={"name": "description"})
            desc_text = meta_desc.get("content", "").strip() if meta_desc else ""
            # 提取前几段文本
            paragraphs = soup.find_all("p")[:3]
            para_text = " ".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])[:500]
            
            return f"{title_text}\n{desc_text}\n{para_text}"
    except Exception as e:
        print(f"[WebSearch] 获取页面内容失败 {url}: {e}")
    return ""

def _web_search_name_type(name: str, max_results: int = 5) -> Tuple[Optional[bool], Optional[str]]:
    """
    通过网络搜索判断名称是数据集还是模型。
    使用 LLM 分析搜索结果内容，而不是简单的关键词匹配。
    
    Returns:
        (is_dataset, confidence_reason)
        is_dataset: True=数据集, False=模型, None=无法确定
        confidence_reason: 判断依据的文本片段
    """
    # 检查缓存
    cache_key = name.lower().strip()
    if cache_key in _web_search_cache:
        cached = _web_search_cache[cache_key]
        # 检查缓存是否过期（7天）
        cache_time = cached.get("timestamp", 0)
        if time.time() - cache_time < 7 * 24 * 3600:
            return cached.get("is_dataset"), cached.get("reason")
    
    # 构建搜索查询
    search_queries = [
        f'"{name}" dataset OR benchmark',
        f'"{name}" model OR architecture',
    ]
    
    search_results_text = []
    
    try:
        # 使用 Google 搜索（如果可用）
        if GOOGLE_SEARCH_AVAILABLE:
            for query in search_queries:
                try:
                    results = list(google_search(query, num_results=min(5, max_results), lang="en"))
                    for url in results:
                        # 获取页面内容
                        content = _fetch_search_result_content(url)
                        if content:
                            search_results_text.append(f"URL: {url}\nContent: {content[:800]}\n---")
                    
                    time.sleep(1)  # 避免请求过快
                except Exception as e:
                    print(f"[WebSearch] Google搜索失败: {e}")
                    continue
        
        # 备用方案：使用 DuckDuckGo（如果可用）
        elif REQUESTS_AVAILABLE:
            search_url = f"https://html.duckduckgo.com/html/?q={name}+dataset+OR+benchmark+OR+model"
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                response = requests.get(search_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    # 提取搜索结果
                    results = soup.find_all("div", class_="result")[:max_results]
                    for result in results:
                        title_elem = result.find("a", class_="result__a")
                        snippet_elem = result.find("a", class_="result__snippet")
                        if title_elem:
                            title = title_elem.get_text().strip()
                            url = title_elem.get("href", "")
                            snippet = snippet_elem.get_text().strip() if snippet_elem else ""
                            search_results_text.append(f"URL: {url}\nTitle: {title}\nSnippet: {snippet}\n---")
            except Exception as e:
                print(f"[WebSearch] DuckDuckGo搜索失败: {e}")
    
    except Exception as e:
        print(f"[WebSearch] 网络搜索出错: {e}")
    
    # 如果没有搜索结果，返回无法确定
    if not search_results_text:
        reason = "无法获取网络搜索结果"
        _web_search_cache[cache_key] = {
            "is_dataset": None,
            "reason": reason,
            "timestamp": time.time()
        }
        _save_search_cache()
        return None, reason
    
        # 使用 LLM 分析搜索结果
    try:
        # 构建提示词
        search_context = "\n".join(search_results_text[:10])  # 限制长度
        
        analysis_prompt = f"""请分析以下网络搜索结果，判断 "{name}" 是数据集（benchmark/dataset）还是模型（model/architecture）。

搜索结果：
{search_context}

请根据搜索结果的内容（标题、描述、正文）来判断。注意：
- 数据集通常包含：benchmark, dataset, evaluation, evaluation set, test set 等关键词
- 模型通常包含：model, architecture, proposes, introduces, presents 等关键词
- 如果搜索结果明确说明是数据集或模型，请相信搜索结果
- 如果搜索结果提到在某个数据集上评估模型，那么该名称是模型
- 如果搜索结果提到某个数据集用于评估，那么该名称是数据集

请以 JSON 格式返回：
{{
    "type": "dataset" 或 "model" 或 "unknown",
    "confidence": "high" 或 "medium" 或 "low",
    "reason": "判断依据（引用搜索结果中的关键信息）"
}}"""

        # 使用 LLM 分析（需要先获取 model 实例）
        # 注意：这里需要延迟导入，因为 model 在文件末尾才定义
        from google.adk.models.lite_llm import LiteLlm
        # 与主 Agent 保持一致的提供商选择
        _provider = os.getenv("LLM_PROVIDER", "gpt").strip().lower()
        if _provider == "deepseek":
            analysis_model = LiteLlm(model="deepseek/deepseek-chat")
        elif _provider in ("gpt", "openai"):
            analysis_model = LiteLlm(model="openai/gpt-4o-mini")
        elif _provider == "qwen":
            analysis_model = LiteLlm(model="qwen/qwen-plus")
        elif _provider == "gemini":
            analysis_model = LiteLlm(model="gemini/gemini-2.5-flash")
        else:
            analysis_model = LiteLlm(model="openai/gpt-4o-mini")
        
        llm_response = analysis_model.generate_content(analysis_prompt)
        response_text = llm_response.text if hasattr(llm_response, 'text') else str(llm_response)
        
        # 解析 LLM 响应
        import json as json_module
        try:
            # 尝试提取 JSON
            json_match = re.search(r'\{[^{}]*"type"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                result = json_module.loads(json_match.group())
                result_type = result.get("type", "unknown")
                confidence = result.get("confidence", "low")
                reason = result.get("reason", "LLM分析结果")
                
                is_dataset = None
                if result_type == "dataset":
                    is_dataset = True
                elif result_type == "model":
                    is_dataset = False
                
                # 保存到缓存
                _web_search_cache[cache_key] = {
                    "is_dataset": is_dataset,
                    "reason": f"[LLM分析] {reason} (置信度: {confidence})",
                    "timestamp": time.time()
                }
                _save_search_cache()
                
                return is_dataset, f"[LLM分析] {reason} (置信度: {confidence})"
        except Exception as e:
            print(f"[WebSearch] 解析LLM响应失败: {e}")
            print(f"[WebSearch] LLM响应: {response_text[:500]}")
    
    except Exception as e:
        print(f"[WebSearch] LLM分析失败: {e}")
    
    # 如果 LLM 分析失败，回退到简单的关键词分析
    all_text = "\n".join(search_results_text).lower()
    dataset_keywords = ["dataset", "benchmark", "evaluation set", "test set"]
    model_keywords = ["model", "architecture", "proposes", "introduces", "presents"]
    
    dataset_count = sum(1 for kw in dataset_keywords if kw in all_text)
    model_count = sum(1 for kw in model_keywords if kw in all_text)
    
    is_dataset = None
    reason = None
    
    if dataset_count > model_count * 1.5:
        is_dataset = True
        reason = f"关键词分析：数据集相关关键词出现 {dataset_count} 次，模型相关关键词出现 {model_count} 次"
    elif model_count > dataset_count * 1.5:
        is_dataset = False
        reason = f"关键词分析：模型相关关键词出现 {model_count} 次，数据集相关关键词出现 {dataset_count} 次"
    else:
        reason = f"关键词分析：无法确定（数据集: {dataset_count}, 模型: {model_count}）"
    
    # 保存到缓存
    _web_search_cache[cache_key] = {
        "is_dataset": is_dataset,
        "reason": reason,
        "timestamp": time.time()
    }
    _save_search_cache()
    
    return is_dataset, reason

def _analyze_query_context(text: str, candidate_name: str) -> Tuple[bool, bool]:
    """
    分析查询上下文，判断候选名称是数据集还是模型。
    
    Returns:
        (is_likely_dataset, is_likely_model)
        如果两者都为 False，表示无法确定
    """
    text_lower = text.lower()
    candidate_lower = candidate_name.lower()
    
    # 检查明确的关键词
    explicit_dataset_keywords = ["数据集", "dataset", "benchmark", "基准", "evaluated on", "tested on"]
    explicit_model_keywords = ["模型", "model", "architecture", "架构", "proposes", "introduces"]
    
    has_explicit_dataset = any(kw in text_lower for kw in explicit_dataset_keywords)
    has_explicit_model = any(kw in text_lower for kw in explicit_model_keywords)
    
    # 检查查询模式
    dataset_score = 0
    model_score = 0
    
    # 模式1: "X 上的 SOTA" 或 "SOTA on X" → X 是数据集
    if re.search(rf"(?:sota|state[- ]of[- ]the[- ]art|performance|accuracy|score)\s+(?:on|在)\s+{re.escape(candidate_lower)}", text_lower):
        dataset_score += 3
    if re.search(rf"{re.escape(candidate_lower)}\s+(?:上的|上|on)\s+(?:sota|state[- ]of[- ]the[- ]art)", text_lower):
        dataset_score += 3
    
    # 模式2: "在 X 上" 或 "on X" → X 是数据集
    if re.search(rf"(?:在|on|evaluated\s+on|tested\s+on|benchmark)\s+{re.escape(candidate_lower)}", text_lower):
        dataset_score += 2
    
    # 模式3: "X 数据集" 或 "X dataset" → X 是数据集
    if re.search(rf"{re.escape(candidate_lower)}\s+(?:数据集|dataset|benchmark|基准)", text_lower):
        dataset_score += 3
    
    # 模式4: "X 模型" 或 "X model" → X 是模型
    if re.search(rf"{re.escape(candidate_lower)}\s+(?:模型|model|architecture|架构)", text_lower):
        model_score += 3
    
    # 模式5: "proposes/introduces X" → X 是模型
    if re.search(rf"(?:proposes|introduces|presents|develops|designs)\s+{re.escape(candidate_lower)}", text_lower):
        model_score += 2
    
    # 模式6: "X achieves/obtains" → X 是模型
    if re.search(rf"{re.escape(candidate_lower)}\s+(?:achieves|obtains|reaches|gets)", text_lower):
        model_score += 2
    
    # 模式7: "找 X 上的" 或 "find SOTA on X" → X 是数据集
    if re.search(rf"(?:找|find|search|query|get)\s+(?:.*?)?(?:上的|上|on)\s+{re.escape(candidate_lower)}", text_lower):
        dataset_score += 2
    if re.search(rf"(?:找|find|search|query|get)\s+{re.escape(candidate_lower)}\s+(?:上的|上)", text_lower):
        dataset_score += 2
    
    # 明确关键词的权重
    if has_explicit_dataset and candidate_lower in text_lower:
        dataset_score += 1
    if has_explicit_model and candidate_lower in text_lower:
        model_score += 1
    
    # 判断结果
    is_likely_dataset = dataset_score > model_score and dataset_score > 0
    is_likely_model = model_score > dataset_score and model_score > 0
    
    return is_likely_dataset, is_likely_model

def _detect_datasets(text: str) -> List[str]:
    """
    从文本中检测数据集名称（用于论文摘要分析）。
    基于上下文智能判断，而不是依赖黑名单。
    """
    tl = text.lower()
    found: List[str] = []
    
    for name, pats in _DATASET_PATTERNS.items():
        # 检查是否匹配模式
        if not any(re.search(p, tl) for p in pats):
            continue
        
        # 分析上下文，判断这个名称更可能是数据集还是模型
        is_likely_dataset, is_likely_model = _analyze_query_context(text, name)
        
        # 如果明确是模型，跳过
        if is_likely_model and not is_likely_dataset:
            continue
        
        # 如果明确是数据集，或者无法确定但出现在数据集上下文中，添加
        if is_likely_dataset or (not is_likely_model and any(kw in tl for kw in ["on ", "evaluated", "benchmark", "dataset"])):
            found.append(name)
    
    return sorted(list(set(found)))

def _extract_metric_with_dataset(title: str, summary: str) -> Dict[str, Any]:
    text = f"{title}\n{summary}"
    pat, score = _extract_metric(text)
    datasets = _detect_datasets(text)
    return {"metric_pattern": pat, "metric_score": (score if score >= 0 else None), "datasets": datasets}

def _score_key_for_dataset(metric_score: Optional[float], datasets: List[str]) -> float:
    # 若能识别数据集，直接使用指标分数；否则稍微降权
    if metric_score is None:
        return -1.0
    return metric_score if datasets else (metric_score * 0.95)

def get_latest_sota(
    benchmark: Optional[str] = None, 
    query: Optional[str] = None,
    window_days: Optional[int] = None, 
    max_results: Optional[int] = None, 
    scope: Optional[str] = None, 
    constraints: Optional[Dict[str, Any]] = None
) -> str:
    """
    查询最新 SOTA 模型（支持自然语言查询和直接参数两种方式）。
    
    方式1 - 自然语言查询（推荐）：
        传入 query 参数，例如：
        - query="找 GOT-10k 上纯监督、不要自监督、近一年最新的 SOTA，且不使用额外数据"
        - query="RT-1 数据集上最新的 SOTA 模型"
        - query="VLA 领域最近半年的 SOTA"
    
    方式2 - 直接参数：
        benchmark: 基准名称，例如 "RT-1", "GOT-10k" 等
        window_days: 搜索时间窗口（天数），默认 365 天
        max_results: 最大结果数，默认 100
        scope: 可选 "overall"（默认）、"self-supervised"、"supervised"、"semi-supervised"、"weakly-supervised"、"unsupervised"、"zero-shot"、"few-shot"
        constraints: 可选约束条件，格式为 {data_regime:[], modality:[], tricks:[], resources:[], require_dataset:bool}
    
    如果同时提供了 query 和 benchmark，query 优先。
    如果只提供了一个字符串参数且长度较长（>10字符），则自动识别为自然语言查询。
    """
    # 智能判断：如果只传入了一个位置参数且是长字符串，或提供了 query，则使用自然语言解析
    use_nl = False
    nl_query = None
    
    # 检查是否提供了 query 参数
    if query:
        nl_query = query
        use_nl = True
    # 如果只提供了 benchmark 且是长字符串，可能也是自然语言
    elif benchmark and len(benchmark) > 30 and not all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.' for c in benchmark):
        nl_query = benchmark
        use_nl = True
    # 如果 benchmark 很短或看起来像是数据集名称，且有其他参数，则使用直接参数模式
    elif benchmark:
        use_nl = False
    else:
        return json.dumps({
            "error": "请提供 benchmark 参数或 query 参数（自然语言查询）",
            "example_nl": "query='找 GOT-10k 上最新的 SOTA 模型'",
            "example_direct": "benchmark='GOT-10k', window_days=365"
        }, ensure_ascii=False, indent=2)
    
    # 自然语言模式：解析并调用核心函数
    if use_nl and nl_query:
        # 检查消息中是否包含过滤模式提示
        relaxed_mode = "宽松模式" in nl_query or "[系统提示：当前过滤模式为宽松模式" in nl_query
        
        # 检测用户是否要求"最强"、"性能最好"等
        sort_by_performance = _nl_detect_best_performance(nl_query)
        
        parsed_benchmark = _nl_detect_benchmark(nl_query)
        include_scopes, exclude_scopes, strict_scope = _nl_detect_scopes(nl_query)
        parsed_constraints = _nl_detect_constraints(nl_query)
        if include_scopes:
            parsed_constraints["include_scopes"] = sorted(list(include_scopes))
        if exclude_scopes:
            parsed_constraints["exclude_scopes"] = sorted(list(exclude_scopes))
        if strict_scope:
            parsed_constraints["strict_scope"] = True
        
        # 如果用户要求"最强"，排除轻量化模型
        if sort_by_performance:
            if "forbidden_terms" not in parsed_constraints:
                parsed_constraints["forbidden_terms"] = []
            # 添加轻量化模型关键词到禁止列表
            lightweight_keywords = ["轻量化", "lightweight", "efficient", "mobile", "lite", "tiny", "small model", "compact"]
            for kw in lightweight_keywords:
                if kw not in parsed_constraints["forbidden_terms"]:
                    parsed_constraints["forbidden_terms"].append(kw)
        
        parsed_window_days = _nl_detect_window_days(nl_query, 365) if nl_query else 365
        parsed_max_results = 150  # 自然语言查询默认更多结果
        parsed_scope = "overall"
        
        # 使用解析后的参数调用核心函数
        return _get_latest_sota_core(
            benchmark=parsed_benchmark,
            window_days=parsed_window_days,
            max_results=parsed_max_results,
            scope=parsed_scope,
            constraints=parsed_constraints,
            relaxed_mode=relaxed_mode,
            sort_by_performance=sort_by_performance
        )
    
    # 直接参数模式：使用提供的参数，如果未提供则使用默认值
    if not benchmark:
        return json.dumps({
            "error": "请提供 benchmark 参数或 query 参数（自然语言查询）",
            "example_nl": "query='找 GOT-10k 上最新的 SOTA 模型'",
            "example_direct": "benchmark='GOT-10k', window_days=365"
        }, ensure_ascii=False, indent=2)
    
    final_window_days = window_days if window_days is not None else 365
    final_max_results = max_results if max_results is not None else 100
    final_scope = scope if scope is not None else "overall"
    final_constraints = constraints if constraints is not None else {}
    
    # 检查 constraints 中是否有 relaxed_mode 标志
    relaxed_mode = final_constraints.pop("relaxed_mode", False) if isinstance(final_constraints, dict) else False
    sort_by_performance = final_constraints.pop("sort_by_performance", False) if isinstance(final_constraints, dict) else False
    
    return _get_latest_sota_core(
        benchmark=benchmark,
        window_days=final_window_days,
        max_results=final_max_results,
        scope=final_scope,
        constraints=final_constraints,
        relaxed_mode=relaxed_mode,
        sort_by_performance=sort_by_performance
    )

def _get_latest_sota_core(benchmark: str, window_days: int = 365, max_results: int = 100, scope: str = "overall", constraints: Optional[Dict[str, Any]] = None, relaxed_mode: bool = False, sort_by_performance: bool = False) -> str:
    """
    核心 SOTA 查询逻辑（内部函数）。
    根据关键词检索近 window_days 天的论文，启发式识别 SOTA，返回包含最新 SOTA 的 JSON。
    scope 可选：overall（默认）、self-supervised、supervised、semi-supervised、weakly-supervised、unsupervised、zero-shot、few-shot
    constraints 可选：{data_regime:[], modality:[], tricks:[], resources:[], require_dataset:bool}
    relaxed_mode: 如果为 True，在严格过滤无结果时自动放宽约束；如果为 False，严格匹配所有约束
    sort_by_performance: 如果为 True，按性能指标排序（优先返回性能最好的）；如果为 False，按时间排序（优先返回最新的）
    """
    client = arxiv.Client()
    query = f'ti:"{benchmark}" OR abs:"{benchmark}"'
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    now = datetime.utcnow().date()
    candidates: List[Dict[str, Any]] = []
    constraints = constraints or {}
    # 兼容：若传入 scope 且非 overall，等价于 include_scopes=[scope]
    include_scopes: Set[str] = set(map(str, (constraints.get("include_scopes") or [])))
    if scope and scope != "overall":
        include_scopes.add(scope)
    exclude_scopes: Set[str] = set(map(str, (constraints.get("exclude_scopes") or [])))
    strict_scope: bool = bool(constraints.get("strict_scope", False))
    dataset_filter: Set[str] = set(map(str, (constraints.get("datasets") or [])))
    require_dataset = bool(constraints.get("require_dataset", False))

    try:
        search_results = _fetch_arxiv_results(client, search)
    except RuntimeError as err:
        return json.dumps({
            "benchmark": benchmark,
            "scope": scope,
            "constraints": constraints,
            "sota": None,
            "message": str(err)
        }, ensure_ascii=False, indent=2)

    for p in search_results:
        if not p.published:
            continue
        if (now - p.published.date()) > timedelta(days=window_days):
            continue
        d = _paper_to_dict(p)
        ctz = _extract_constraints(d["title"], d["summary"] or "")
        d.update(ctz)
        scopes = set(ctz.get("scopes") or [])
        # 作用域过滤（包含/排除/严格）
        if include_scopes:
            if strict_scope:
                # 严格模式：论文的 scopes 必须是 include_scopes 的子集，且非空
                if not scopes or not scopes.issubset(include_scopes):
                    continue
            else:
                # 宽松模式：与 include_scopes 有交集即可
                if not (scopes & include_scopes):
                    continue
        if exclude_scopes and (scopes & exclude_scopes):
            continue
        # 其它 constraints 过滤（若用户提供）
        def match_list(key: str) -> bool:
            want = set(constraints.get(key) or [])
            if not want:
                return True
            have = set(ctz.get(key) or [])
            return bool(want & have)  # 交集匹配
        if not all([match_list("data_regime"), match_list("modality"), match_list("tricks"), match_list("resources")]):
            continue
        # 必含/必排术语（标题+摘要）
        required_terms = list(map(str, (constraints.get("required_terms") or [])))
        forbidden_terms = list(map(str, (constraints.get("forbidden_terms") or [])))
        blob = f"{d['title']}\n{d.get('summary') or ''}".lower()
        if required_terms and not all(term.lower() in blob for term in required_terms):
            continue
        if forbidden_terms and any(term.lower() in blob for term in forbidden_terms):
            continue

        # 必含/必排术语（标题+摘要）
        required_terms = list(map(str, (constraints.get("required_terms") or [])))
        forbidden_terms = list(map(str, (constraints.get("forbidden_terms") or [])))
        blob = f"{d['title']}\n{d.get('summary') or ''}".lower()
        if required_terms and not all(term.lower() in blob for term in required_terms):
            continue
        if forbidden_terms and any(term.lower() in blob for term in forbidden_terms):
            continue
        met = _extract_metric_with_dataset(d["title"], d["summary"] or "")
        d.update(met)
        if require_dataset and not d["datasets"]:
            continue
        if dataset_filter:
            if not set(d["datasets"] or []) & dataset_filter:
                continue
        is_sota = _looks_like_sota(d["title"], d["summary"])
        # 排序信号：SOTA 线索、指标（结合是否识别出数据集）、显式匹配 scope、时间
        metric_key = _score_key_for_dataset(d.get("metric_score"), d.get("datasets") or [])
        rank_score = (
            1 if is_sota else 0,
            metric_key,
            1 if (scope != "overall" and scope in scopes) else 0,
            p.published,
        )
        d.update({
            "sota_signal": is_sota,
            "rank_score": [rank_score[0], rank_score[1], rank_score[2], str(rank_score[3])],
            "evidence": (d["summary"] or "")[:400],  # 证据片段（摘要前 400 字符）
        })
        candidates.append(d)

    # 如果严格模式下没有结果，且启用了宽松模式，尝试放宽约束
    if not candidates and relaxed_mode:
        print(f"[宽松模式] 严格过滤无结果，尝试放宽约束条件...")
        # 重新搜索，但放宽大部分约束
        candidates = []
        for p in search_results:
            if not p.published:
                continue
            if (now - p.published.date()) > timedelta(days=window_days):
                continue
            d = _paper_to_dict(p)
            ctz = _extract_constraints(d["title"], d["summary"] or "")
            d.update(ctz)
            scopes = set(ctz.get("scopes") or [])
            
            # 宽松模式：只保留最基本的过滤（排除明确不想要的 scope，其他都保留）
            if exclude_scopes and (scopes & exclude_scopes):
                continue
            
            # 宽松模式：只检查 forbidden_terms（明确禁止的），不检查 required_terms
            forbidden_terms = list(map(str, (constraints.get("forbidden_terms") or [])))
            blob = f"{d['title']}\n{d.get('summary') or ''}".lower()
            if forbidden_terms and any(term.lower() in blob for term in forbidden_terms):
                continue
            
            met = _extract_metric_with_dataset(d["title"], d["summary"] or "")
            d.update(met)
            
            is_sota = _looks_like_sota(d["title"], d["summary"])
            metric_key = _score_key_for_dataset(d.get("metric_score"), d.get("datasets") or [])
            rank_score = (
                1 if is_sota else 0,
                metric_key,
                1 if (scope != "overall" and scope in scopes) else 0,
                p.published,
            )
            d.update({
                "sota_signal": is_sota,
                "rank_score": [rank_score[0], rank_score[1], rank_score[2], str(rank_score[3])],
                "evidence": (d["summary"] or "")[:400],
            })
            candidates.append(d)
        
        if candidates:
            print(f"[宽松模式] 放宽约束后找到 {len(candidates)} 个候选结果")
    
    if not candidates:
        mode_msg = "（宽松模式下已尝试放宽约束）" if relaxed_mode else ""
        return json.dumps({"benchmark": benchmark, "scope": scope, "constraints": constraints, "sota": None, "message": f"未检索到近一年可能的 SOTA 论文{mode_msg}"}, ensure_ascii=False, indent=2)

    # 排序策略：根据 sort_by_performance 决定
    if sort_by_performance:
        # 按性能排序：优先考虑性能指标，然后才是时间和 SOTA 信号
        candidates.sort(key=lambda x: (
            (x.get("metric_score") or -1.0),  # 性能指标优先
            1 if x.get("sota_signal") else 0,  # SOTA 信号次之
            (1 if (include_scopes and (set(x.get("scopes") or []) & include_scopes)) else 0),
            datetime.fromisoformat(x["updated"]) if x.get("updated") else datetime.min,
        ), reverse=True)
    else:
        # 按时间排序：SOTA 线索优先，指标越大越好，时间越新越好（默认）
        candidates.sort(key=lambda x: (
            1 if x.get("sota_signal") else 0,
            (x.get("metric_score") or -1.0),
            (1 if (include_scopes and (set(x.get("scopes") or []) & include_scopes)) else 0),
            datetime.fromisoformat(x["updated"]) if x.get("updated") else datetime.min,
        ), reverse=True)

    best = candidates[0]
    result = {
        "benchmark": benchmark,
        "scope": scope,
        "constraints": constraints,
        "sota": {
            "id": best["id"],
            "title": best["title"],
            "published": best["published"],
            "pdf_url": best["pdf_url"],
            "arxiv_url": best.get("arxiv_url"),
            "metric": best.get("metric_score"),
            "metric_pattern": best.get("metric_pattern"),
            "sota_signal": best.get("sota_signal"),
            "scopes": best.get("scopes") or [],
            "datasets": best.get("datasets") or [],
            "evidence": best.get("evidence"),
        },
        "top_candidates": [
            {
                "id": c["id"],
                "title": c["title"],
                "published": c["published"],
                "metric": c.get("metric_score"),
                "sota_signal": c.get("sota_signal"),
                "scopes": c.get("scopes") or [],
                "datasets": c.get("datasets") or [],
                "pdf_url": c["pdf_url"],
                "arxiv_url": c.get("arxiv_url"),
                "evidence": (c.get("summary") or "")[:200],
            } for c in candidates[:5]
        ]
    }
    return json.dumps(result, ensure_ascii=False, indent=2)

def list_recent_papers(benchmark: str, limit: int = 10, window_days: int = 180, scope: str = "overall", constraints: Optional[Dict[str, Any]] = None) -> str:
    """
    返回近期（window_days）相关论文列表（按时间倒序），包含基本元信息、启发式指标、范围与约束标签。
    scope 可选：overall（默认）、self-supervised、supervised、semi-supervised、weakly-supervised、unsupervised、zero-shot、few-shot
    constraints 可选：同 get_latest_sota
    """
    client = arxiv.Client()
    query = f'ti:"{benchmark}" OR abs:"{benchmark}"'
    search = arxiv.Search(
        query=query,
        max_results=max(100, limit),
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    now = datetime.utcnow().date()
    items: List[Dict[str, Any]] = []
    constraints = constraints or {}
    include_scopes: Set[str] = set(map(str, (constraints.get("include_scopes") or [])))
    if scope and scope != "overall":
        include_scopes.add(scope)
    exclude_scopes: Set[str] = set(map(str, (constraints.get("exclude_scopes") or [])))
    strict_scope: bool = bool(constraints.get("strict_scope", False))
    dataset_filter: Set[str] = set(map(str, (constraints.get("datasets") or [])))
    require_dataset = bool(constraints.get("require_dataset", False))

    try:
        search_results = _fetch_arxiv_results(client, search)
    except RuntimeError as err:
        return json.dumps({
            "benchmark": benchmark,
            "scope": scope,
            "constraints": constraints,
            "message": str(err)
        }, ensure_ascii=False, indent=2)

    for p in search_results:
        if not p.published:
            continue
        if (now - p.published.date()) > timedelta(days=window_days):
            continue
        d = _paper_to_dict(p)
        ctz = _extract_constraints(d["title"], d["summary"] or "")
        d.update(ctz)
        scopes = set(ctz.get("scopes") or [])
        # 作用域过滤
        if include_scopes:
            if strict_scope:
                if not scopes or not scopes.issubset(include_scopes):
                    continue
            else:
                if not (scopes & include_scopes):
                    continue
        if exclude_scopes and (scopes & exclude_scopes):
            continue
        def match_list(key: str) -> bool:
            want = set(constraints.get(key) or [])
            if not want:
                return True
            have = set(ctz.get(key) or [])
            return bool(want & have)
        if not all([match_list("data_regime"), match_list("modality"), match_list("tricks"), match_list("resources")]):
            continue
        met = _extract_metric_with_dataset(d["title"], d["summary"] or "")
        d.update(met)
        if require_dataset and not d["datasets"]:
            continue
        if dataset_filter:
            if not set(d["datasets"] or []) & dataset_filter:
                continue
        d["sota_signal"] = _looks_like_sota(d["title"], d["summary"])
        d["evidence"] = (d["summary"] or "")[:200]
        items.append(d)

    items.sort(key=lambda d: (d.get("metric_score") or -1.0, d.get("published") or ""), reverse=True)
    return json.dumps(items[:limit], ensure_ascii=False, indent=2)

# ------------------- 自然语言解析 → benchmark / constraints / 时间窗口 -------------------
_CN_SCOPE_SYNONYMS = {
    "self-supervised": ["自监督", "自我监督"],
    "supervised": ["监督学习", "有监督", "纯监督", "完全监督", "只看监督"],
    "reinforcement": ["强化学习", "增强学习", "RL"],
    "semi-supervised": ["半监督"],
    "weakly-supervised": ["弱监督"],
    "unsupervised": ["无监督"],
    "zero-shot": ["零样本"],
    "few-shot": ["小样本", "少样本", "单样本", "一-shot", "one-shot"],
}

_CN_DATA_REGIME_SYNONYMS = {
    "no-extra-data": ["不使用额外数据", "无额外数据", "只用官方数据", "仅训练集"],
    "extra-data": ["使用额外数据", "外部数据", "额外数据", "网络规模", "web规模"],
    "pretrained": ["预训练", "预训练模型", "foundation"],
    "distilled": ["蒸馏", "distill"],
}

_CN_TRICKS_SYNONYMS = {
    "tta": ["测试时增强", "TTA"],
    "ensemble": ["集成", "ensemble"],
    "prompting": ["提示工程", "prompt", "prompting", "提示微调"],
}

_CN_RES_SYNONYMS = {
    "realtime": ["实时", "实时性", "fps"],
    "lightweight": ["轻量", "小模型", "tiny"],
}

_CN_TIME_PATTERNS = [
    (r"(近|最近)(\d{1,3})天", "days"),
    (r"(近|最近)(\d{1,2})个月", "months"),
    (r"(近|最近)(\d{1,2})年", "years"),
    (r"(过去|最近)(\d{1,3})\s*days?", "days"),
    (r"(过去|最近)(\d{1,2})\s*months?", "months"),
    (r"(过去|最近)(\d{1,2})\s*years?", "years"),
]

def _nl_detect_benchmark(text: str) -> str:
    """
    从自然语言文本中智能检测数据集名称。
    基于查询模式、上下文和网络搜索进行判断。
    """
    text_lower = text.lower()
    
    # 收集所有匹配的候选名称
    candidates: List[Tuple[str, int]] = []  # (name, match_position)
    
    for ds in _DATASET_PATTERNS.keys():
        # 检查是否在文本中出现
        match = re.search(rf"\b{re.escape(ds)}\b", text, flags=re.I)
        if match:
            candidates.append((ds, match.start()))
    
    if not candidates:
        return text.strip()
    
    # 如果只有一个候选，分析上下文
    if len(candidates) == 1:
        candidate_name = candidates[0][0]
        is_likely_dataset, is_likely_model = _analyze_query_context(text, candidate_name)
        
        # 如果明确是数据集，返回
        if is_likely_dataset:
            return candidate_name
        # 如果明确是模型，跳过（返回原始文本，让后续处理）
        if is_likely_model:
            return text.strip()
        
        # 如果上下文分析无法确定，使用网络搜索作为回退
        if not is_likely_dataset and not is_likely_model:
            web_is_dataset, web_reason = _web_search_name_type(candidate_name)
            if web_is_dataset is True:
                print(f"[WebSearch] 网络搜索确认 '{candidate_name}' 是数据集: {web_reason}")
                return candidate_name
            elif web_is_dataset is False:
                print(f"[WebSearch] 网络搜索确认 '{candidate_name}' 是模型: {web_reason}")
                return text.strip()
        
        # 如果查询模式暗示是数据集查询（如"找 X 上的 SOTA"），返回候选
        if any(kw in text_lower for kw in ["找", "find", "search", "sota", "performance", "accuracy"]):
            return candidate_name
        # 否则返回原始文本
        return text.strip()
    
    # 如果有多个候选，选择最可能是数据集的
    best_candidate = None
    best_score = -1
    
    for candidate_name, _ in candidates:
        is_likely_dataset, is_likely_model = _analyze_query_context(text, candidate_name)
        
        if is_likely_dataset and not is_likely_model:
            # 计算得分（位置越靠前，得分越高）
            score = 10
            if any(kw in text_lower for kw in ["数据集", "dataset", "benchmark"]):
                score += 5
            if best_score < score:
                best_score = score
                best_candidate = candidate_name
        elif not is_likely_model:
            # 如果上下文无法确定，尝试网络搜索
            if not is_likely_dataset:
                web_is_dataset, _ = _web_search_name_type(candidate_name)
                if web_is_dataset is True:
                    score = 8
                elif web_is_dataset is False:
                    continue  # 明确是模型，跳过
                else:
                    score = 5
            else:
                score = 5
            
            if best_score < score:
                best_score = score
                best_candidate = candidate_name
    
    if best_candidate:
        return best_candidate
    
    # 如果无法确定，返回第一个候选（保持向后兼容）
    return candidates[0][0]

def _nl_detect_scopes(text: str) -> Tuple[Set[str], Set[str], bool]:
    t = text.lower()
    include: Set[str] = set()
    exclude: Set[str] = set()
    strict = False
    for scope, syns in _CN_SCOPE_SYNONYMS.items():
        if any(s in text for s in syns) or scope in t:
            include.add(scope)
    if any(kw in text for kw in ["不要自监督", "排除自监督", "不含自监督", "exclude self"]):
        exclude.add("self-supervised")
    if any(kw in text for kw in ["不要强化学习", "排除强化学习", "不含强化学习", "exclude rl", "exclude reinforcement"]):
        exclude.add("reinforcement")
    if "只看监督" in text or "纯监督" in text or "strict supervised" in t:
        include.add("supervised")
        strict = True
    return include, exclude, strict

def _nl_detect_constraints(text: str) -> Dict[str, Any]:
    c: Dict[str, Any] = {"data_regime": [], "modality": [], "tricks": [], "resources": []}
    def match_syn(syn_map: Dict[str, List[str]], key: str):
        for k, syns in syn_map.items():
            if any(s in text for s in syns):
                c[key].append(k)
    match_syn(_CN_DATA_REGIME_SYNONYMS, "data_regime")
    match_syn(_CN_TRICKS_SYNONYMS, "tricks")
    match_syn(_CN_RES_SYNONYMS, "resources")
    # 通用“必排/必含”术语解析（泛化，不依赖预置列表）
    def _clean_terms(terms):
        out = []
        for t in terms:
            t = re.sub(r"^[，。,.；;、\s]+|[，。,.；;、\s]+$", "", t)
            t = t.strip()
            if 1 <= len(t) <= 32:
                out.append(t)
        return list(dict.fromkeys(out))
    forbidden = []
    # 中文：不要X / 不含X / 排除X
    for m in re.findall(r"(?:不要|不含|排除)([^\s，。,.；;、]{1,32})", text):
        forbidden.append(m)
    # 英文：exclude X / without X / no X
    for m in re.findall(r"(?:exclude|without|no)\s+([A-Za-z0-9\-\+_\/ ]{1,32})", text, flags=re.I):
        forbidden.append(m.strip())
    required = []
    # 中文：必须X / 只要X / 包含X
    for m in re.findall(r"(?:必须|只要|包含)([^\s，。,.；;、]{1,32})", text):
        required.append(m)
    # 英文：must contain X / require X / with X
    for m in re.findall(r"(?:must\s+contain|require|with)\s+([A-Za-z0-9\-\+_\/ ]{1,32})", text, flags=re.I):
        required.append(m.strip())
    forbidden = _clean_terms(forbidden)
    required = _clean_terms(required)
    if forbidden:
        c["forbidden_terms"] = forbidden
    if required:
        c["required_terms"] = required
    # 检测数据集（基于上下文智能判断）
    datasets = []
    for ds in _DATASET_PATTERNS.keys():
        if not re.search(rf"\b{re.escape(ds)}\b", text, flags=re.I):
            continue
        
        # 分析上下文，判断这个名称更可能是数据集还是模型
        is_likely_dataset, is_likely_model = _analyze_query_context(text, ds)
        
        # 如果明确是模型，跳过
        if is_likely_model and not is_likely_dataset:
            continue
        
        # 如果明确是数据集，或者无法确定但查询模式暗示是数据集查询，添加
        if is_likely_dataset or (not is_likely_model):
            datasets.append(ds)
    
    if datasets:
        c["datasets"] = sorted(list(set(datasets)))
        c["require_dataset"] = True
    if any(k in text for k in ["RGB", "rgb", "只用RGB", "纯RGB"]):
        c["modality"].append("rgb")
    c["data_regime"] = sorted(list(set(c["data_regime"])))
    c["tricks"] = sorted(list(set(c["tricks"])))
    c["resources"] = sorted(list(set(c["resources"])))
    return c

def _nl_detect_best_performance(text: str) -> bool:
    """
    检测用户是否要求"最强"、"性能最好"等关键词
    返回 True 表示应该按性能排序，False 表示按时间排序
    """
    text_lower = text.lower()
    performance_keywords = [
        "最强", "性能最好", "性能最高", "最高分", "最好", "最佳性能",
        "best performance", "highest", "strongest", "top performance",
        "最高指标", "最好结果", "最佳模型", "性能最优"
    ]
    return any(kw in text_lower for kw in performance_keywords)

def check_name_type(name: str, use_web_search: bool = True) -> str:
    """
    检查一个名称是数据集（benchmark）还是模型。
    主要通过网络搜索和 LLM 分析来判断，不依赖硬编码列表。
    
    Args:
        name: 要检查的名称（如 "RT-1", "GOT-10k" 等）
        use_web_search: 是否使用网络搜索（默认 True，强烈推荐）
    
    Returns:
        JSON 字符串，包含名称类型和判断依据
    """
    name = name.strip()
    if not name:
        return json.dumps({
            "name": name,
            "type": "unknown",
            "reason": "名称为空"
        }, ensure_ascii=False, indent=2)
    
    # 优先使用网络搜索和 LLM 分析
    if use_web_search:
        print(f"[check_name_type] 正在通过网络搜索和 LLM 分析 '{name}' 的类型...")
        web_is_dataset, web_reason = _web_search_name_type(name)
        
        if web_is_dataset is True:
            return json.dumps({
                "name": name,
                "type": "dataset",
                "reason": web_reason,
                "confidence": "high" if "LLM分析" in web_reason else "medium",
                "method": "web_search_llm_analysis"
            }, ensure_ascii=False, indent=2)
        elif web_is_dataset is False:
            return json.dumps({
                "name": name,
                "type": "model",
                "reason": web_reason,
                "confidence": "high" if "LLM分析" in web_reason else "medium",
                "method": "web_search_llm_analysis"
            }, ensure_ascii=False, indent=2)
        else:
            # 网络搜索无法确定，回退到已知列表
            print(f"[check_name_type] 网络搜索无法确定，回退到已知列表检查...")
    
    # 回退方案：检查已知列表（仅作为最后手段）
    is_in_dataset_list = name in _DATASET_PATTERNS
    
    if is_in_dataset_list:
        return json.dumps({
            "name": name,
            "type": "dataset",
            "reason": f"{name} 在已知数据集列表中（但建议使用网络搜索验证）",
            "confidence": "medium",
            "method": "known_list_fallback"
        }, ensure_ascii=False, indent=2)
    
    # 无法确定
    return json.dumps({
        "name": name,
        "type": "unknown",
        "reason": "无法确定类型：网络搜索无明确结果，且不在已知列表中。建议用户提供更多上下文信息。",
        "confidence": "low",
        "method": "none"
    }, ensure_ascii=False, indent=2)

def _nl_detect_window_days(text: str, default_days: int = 365) -> int:
    for pat, unit in _CN_TIME_PATTERNS:
        m = re.search(pat, text, flags=re.I)
        if m:
            n = int(m.group(2))
            if unit == "days":
                return max(1, n)
            if unit == "months":
                return max(1, n * 30)
            if unit == "years":
                return max(1, n * 365)
    if any(k in text for k in ["最新", "近期", "最近", "这半年"]):
        return 180
    return default_days


def recent_by_nl(query: str) -> str:
    """
    自然语言查询近期论文列表（支持中文），自动解析限制条件。
    """
    benchmark = _nl_detect_benchmark(query)
    include_scopes, exclude_scopes, strict_scope = _nl_detect_scopes(query)
    constraints = _nl_detect_constraints(query)
    if include_scopes:
        constraints["include_scopes"] = sorted(list(include_scopes))
    if exclude_scopes:
        constraints["exclude_scopes"] = sorted(list(exclude_scopes))
    if strict_scope:
        constraints["strict_scope"] = True
    window_days = _nl_detect_window_days(query, 180)
    return list_recent_papers(
        benchmark=benchmark,
        limit=20,
        window_days=window_days,
        scope="overall",
        constraints=constraints
    )

def list_common_benchmarks(domain: str = "vla", include_sota: bool = False) -> str:
    """
    快速列出常见 Benchmark 列表（不查询 SOTA，速度很快）。
    
    Args:
        domain: 领域名称，可选 "vla" (Vision-Language-Action), "tracking" 等，默认为 "vla"
        include_sota: 是否同时查询 SOTA（会显著增加耗时），默认为 False
        
    Returns:
        JSON 字符串，包含 Benchmark 列表信息
    """
    # 根据领域选择对应的数据集列表
    if domain.lower() == "vla" or domain.lower() == "vision-language-action":
        benchmarks = _VLA_BENCHMARKS
    elif domain.lower() == "tracking":
        benchmarks = ["LaSOT", "GOT-10k", "OTB", "TrackingNet", "TNL2K"]
    else:
        # 默认返回所有支持的数据集
        benchmarks = _VLA_BENCHMARKS + ["LaSOT", "GOT-10k", "OTB", "TrackingNet", "TNL2K"]
    
    results: Dict[str, Any] = {
        "domain": domain,
        "benchmarks": [{"name": b, "has_sota_info": False} for b in benchmarks],
        "total_benchmarks": len(benchmarks),
        "updated_at": datetime.utcnow().isoformat(),
        "note": "使用 list_common_benchmarks_with_sota 可以查询详细的 SOTA 信息"
    }
    
    if include_sota:
        # 如果需要查询 SOTA，调用详细版本
        return list_common_benchmarks_with_sota(domain=domain, window_days=730, max_results_per_benchmark=30)
    
    return json.dumps(results, ensure_ascii=False, indent=2)

def list_common_benchmarks_with_sota(domain: str = "vla", window_days: int = 730, max_results_per_benchmark: int = 30, 
                                     delay_seconds: float = 3.0, max_benchmarks: Optional[int] = None) -> str:
    """
    列出常见 Benchmark 及其对应的 SOTA 模型（需要较长时间，因为要查询 arXiv）。
    
    Args:
        domain: 领域名称，可选 "vla" (Vision-Language-Action), "tracking" 等，默认为 "vla"
        window_days: 搜索时间窗口（天数），默认 730 天（约2年）
        max_results_per_benchmark: 每个数据集最多检索的论文数量，默认 30（减少以提高速度）
        delay_seconds: 每个 Benchmark 查询之间的延迟（秒），默认 3.0 秒，避免触发 arXiv 速率限制
        max_benchmarks: 最多处理的 Benchmark 数量（None 表示处理全部），默认 None
        
    Returns:
        JSON 字符串，包含每个 Benchmark 及其对应的最新 SOTA 模型信息
    """
    # 根据领域选择对应的数据集列表
    if domain.lower() == "vla" or domain.lower() == "vision-language-action":
        benchmarks = _VLA_BENCHMARKS
    elif domain.lower() == "tracking":
        benchmarks = ["LaSOT", "GOT-10k", "OTB", "TrackingNet", "TNL2K"]
    else:
        # 默认返回所有支持的数据集
        benchmarks = _VLA_BENCHMARKS + ["LaSOT", "GOT-10k", "OTB", "TrackingNet", "TNL2K"]
    
    # 如果设置了最大数量，只处理前面的
    if max_benchmarks is not None and max_benchmarks > 0:
        benchmarks = benchmarks[:max_benchmarks]
    
    results: Dict[str, Any] = {
        "domain": domain,
        "window_days": window_days,
        "benchmarks": [],
        "total_benchmarks": len(benchmarks),
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "updated_at": datetime.utcnow().isoformat(),
        "note": "查询 SOTA 需要时间，请耐心等待。如需快速获取 Benchmark 列表，请使用 list_common_benchmarks"
    }
    
    print(f"正在搜索 {len(benchmarks)} 个 {domain.upper()} Benchmark 的 SOTA 模型（每个查询间隔 {delay_seconds} 秒）...")
    print(f"提示：这个过程可能需要几分钟，请耐心等待...")
    
    for idx, benchmark in enumerate(benchmarks, 1):
        print(f"[{idx}/{len(benchmarks)}] 处理 Benchmark: {benchmark}...")
        try:
            # 获取该 Benchmark 的最新 SOTA
            sota_result = get_latest_sota(
                benchmark=benchmark,
                window_days=window_days,
                max_results=max_results_per_benchmark,
                scope="overall",
                constraints=None
            )
            sota_data = json.loads(sota_result)
            
            benchmark_info = {
                "benchmark": benchmark,
                "sota": sota_data.get("sota"),
                "top_candidates": sota_data.get("top_candidates", [])[:3],  # 只保留前3个候选
                "has_sota": sota_data.get("sota") is not None
            }
            
            # 如果找到了 SOTA，添加模型名称（从标题中提取）
            if benchmark_info["has_sota"] and benchmark_info["sota"]:
                title = benchmark_info["sota"].get("title", "")
                # 尝试从标题中提取模型名称（通常是标题的第一部分或引号中的内容）
                model_name_match = re.search(r'"([^"]+)"|([A-Z][a-z]+[- ]?[A-Z]?[a-z]*[0-9]?)', title)
                if model_name_match:
                    model_name = model_name_match.group(1) or model_name_match.group(2)
                    benchmark_info["model_name"] = model_name
                else:
                    # 如果没有找到引号，尝试使用标题的前几个词
                    words = title.split()
                    if len(words) > 0:
                        benchmark_info["model_name"] = " ".join(words[:5])
                    else:
                        benchmark_info["model_name"] = title[:50]
                results["successful"] += 1
                print(f"  ✓ 找到 SOTA: {benchmark_info.get('model_name', 'N/A')}")
            else:
                print(f"  - 未找到 SOTA")
            
            results["benchmarks"].append(benchmark_info)
            results["processed"] += 1
            
            # 除了最后一个，其他查询后添加延迟
            if idx < len(benchmarks):
                print(f"  等待 {delay_seconds} 秒以避免触发速率限制...")
                time.sleep(delay_seconds)
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ 处理 {benchmark} 时出错: {error_msg[:100]}")
            results["benchmarks"].append({
                "benchmark": benchmark,
                "error": error_msg,
                "has_sota": False
            })
            results["processed"] += 1
            results["failed"] += 1
            
            # 如果是速率限制错误，等待更长时间
            if "429" in error_msg or "rate limit" in error_msg.lower():
                wait_time = delay_seconds * 3
                print(f"  检测到速率限制，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            elif idx < len(benchmarks):
                time.sleep(delay_seconds)
    
    # 统计找到 SOTA 的数量
    results["found_sota_count"] = sum(1 for b in results["benchmarks"] if b.get("has_sota", False))
    
    print(f"\n完成！处理了 {results['processed']} 个 Benchmark，成功找到 {results['found_sota_count']} 个 SOTA，失败 {results['failed']} 个")
    
    return json.dumps(results, ensure_ascii=False, indent=2)

"""根据环境变量选择后端 LLM 提供商与型号，统一封装。
支持：GPT（OpenAI/Azure 视具体配置）、DeepSeek、Qwen（DashScope）、Gemini。
优先从环境变量 `LLM_PROVIDER` 读取（gpt/deepseek/qwen/gemini），否则回退为 gpt。
"""

use_model = os.getenv("LLM_PROVIDER", "gpt").strip().lower()

if use_model == "deepseek":
    model = LiteLlm(model="deepseek/deepseek-chat")
elif use_model == "gpt" or use_model == "openai":
    # 这里使用通用标识，具体由 LiteLlm 配置决定（如 OPENAI_API_KEY / Azure 配置）
    model = LiteLlm(model="openai/gpt-4o-mini")
elif use_model == "qwen":
    # Qwen (DashScope)
    model = LiteLlm(model="qwen/qwen-plus")
elif use_model == "gemini":
    model = LiteLlm(model="gemini/gemini-2.5-flash")
else:
    # 默认回退到 GPT，避免不可用提供商导致崩溃
    model = LiteLlm(model="openai/gpt-4o-mini")


root_agent = Agent(
    name="search_papers_agent",
    model=model,
    description=(
        "Agent to answer questions about papers, benchmarks, and SOTA models. "
        "Can list common benchmarks with their corresponding SOTA models, especially for VLA (Vision-Language-Action) domain."
    ),
    instruction=(
        "You are a helpful agent specialized in answering questions about papers, benchmarks, and State-of-the-Art (SOTA) models. "
        "You can:\n"
        "1. Search for papers on specific topics or benchmarks\n"
        "2. Find the latest SOTA models for specific benchmarks (supports both natural language queries and direct parameters)\n"
        "3. List common benchmarks (especially VLA benchmarks) with their corresponding SOTA models\n"
        "4. Answer questions about research papers and their results\n"
        "5. Run trustworthy SOTA search using Multi-Agent Pipeline (when users need verified, cross-checked results)\n\n"
        "The get_latest_sota function supports two modes:\n"
        "- Natural language mode: Use query parameter, e.g., query='找 GOT-10k 上最新的 SOTA 模型' or query='RT-1 数据集上纯监督的 SOTA'\n"
        "- Direct parameter mode: Use benchmark, window_days, scope, constraints parameters directly\n"
        "The function will automatically detect which mode to use based on the input.\n\n"
        "IMPORTANT: When users ask for the 'strongest', 'best performance', or 'highest score' SOTA model (e.g., '找最强的 SOTA', '性能最好的模型'), "
        "you should pass their exact query to get_latest_sota. The function will automatically:\n"
        "1. Sort results by performance metrics (highest score first) instead of by time\n"
        "2. Exclude lightweight/efficient models (which are optimized for efficiency, not peak performance)\n"
        "3. Return the model with the best performance metrics\n\n"
        "When users need verified, cross-checked SOTA results (e.g., '用可信的方式找 SOTA', '需要验证的 SOTA 结果'), "
        "you can use run_trustworthy_sota_search function. This function uses a Multi-Agent Pipeline that:\n"
        "1. Searches multiple sources (arXiv, Google Scholar)\n"
        "2. Extracts metrics from PDFs\n"
        "3. Normalizes and standardizes metrics\n"
        "4. Verifies and detects conflicts between different sources\n"
        "This is more reliable but slower than get_latest_sota.\n\n"
        "IMPORTANT: If the system message indicates that Vision Model is enabled (use_vision=True), "
        "you should pass use_vision=True and vision_model parameters to run_trustworthy_sota_search. "
        "Vision Model can handle complex tables and charts in PDFs, providing more accurate extraction. "
        "Available vision_model options: 'gpt-4o', 'claude-3-5-sonnet', 'gemini-2.0-flash-exp'.\n\n"
        "When users ask about common benchmarks (e.g., 'VLA常用数据集及其对应的SOTA模型', '常见Benchmark的SOTA模型'), " 
        "you should use the list_common_benchmarks or list_common_benchmarks_with_sota function to provide a comprehensive list of benchmarks "
        "and their corresponding latest SOTA models. Always respond in Chinese (简体中文) when the user asks in Chinese.\n\n"
        "If a user asks for a list of common benchmarks or datasets with their SOTA models, you should directly call "
        "list_common_benchmarks or list_common_benchmarks_with_sota with the appropriate domain parameter (e.g., 'vla' for Vision-Language-Action).\n\n"
        "CRITICAL: When users ask whether a name is a dataset or model (e.g., 'RT-1是数据集吗', 'Is RT-1 a dataset?'), "
        "you MUST use the check_name_type function to get accurate information. DO NOT rely on your own knowledge, as it may be incorrect.\n"
        "Important facts about common names:\n"
        "- RT-1, RT-2, RT-X are MODELS (Robot Transformer series), NOT datasets\n"
        "- PaLM-E, Gato, SayCan, BC-Z are MODELS, NOT datasets\n"
        "- GOT-10k, LaSOT, OTB, TrackingNet, Open-X Embodiment, Bridge, LIBERO, Calvin, ALFRED, Meta-World are DATASETS\n"
        "When in doubt, always use check_name_type function to verify."
    ),
    tools=[
        search_papers,
        extract_info,
        find_papers_by_benchmark,
        get_latest_sota,
        list_recent_papers,
        recent_by_nl,
        list_common_benchmarks,
        list_common_benchmarks_with_sota,
        check_name_type,  # 添加名称类型检查工具
    ] + ([run_trustworthy_sota_search] if PIPELINE_AVAILABLE else []),
)