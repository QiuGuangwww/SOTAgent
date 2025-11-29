"""
Multi-Agent Pipeline for Trustworthy SOTA Tracking
MVP Version: 基础版本，实现核心功能

架构：
- Agent A (Scanner): 搜索论文（arXiv + Google Scholar）
- Agent B (Extractor): 从 PDF 提取文本和简单表格
- Agent C (Normalizer): 指标标准化和转换
- Agent D (Verifier): 冲突检测和验证
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio

# Agent A: Scanner
class ScannerAgent:
    """Agent A: 负责搜索论文（优先 arXiv + 本地榜单缓存；Google Scholar 可选）"""

    def __init__(self, use_scholar: bool = False, scholar_timeout: float = 12.0):
        self.name = "scanner"
        self.use_scholar = use_scholar
        self.scholar_timeout = scholar_timeout

    async def search_arxiv(
        self,
        query: str,
        max_results: int = 10,
        include_terms: Optional[List[str]] = None,
        exclude_terms: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        sort_by_recent: bool = False,
        days_window: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """搜索 arXiv（可配置：包含词、排除词、分类与排序）"""
        try:
            import arxiv, os, asyncio, random
            # 避免企业代理拦截 arXiv
            for k in ("NO_PROXY", "no_proxy"):
                hosts = os.environ.get(k, "")
                hostset = {h.strip() for h in hosts.split(",") if h.strip()}
                hostset.update({"export.arxiv.org", "arxiv.org"})
                os.environ[k] = ",".join(sorted(hostset))

            client = arxiv.Client()

            # 中文查询 → 自动附加英文同义词（例如 多智能体强化学习 → MARL）
            q = (query or "").strip()
            q_lower = q.lower()
            has_cjk = any(ord(ch) > 127 for ch in q)
            extra_terms: List[str] = []
            rl_hint = False
            if has_cjk:
                mapping = {
                    "多智能体": ['"multi-agent"', '"multi agent"'],
                    "强化学习": ['"reinforcement learning"', 'RL'],
                    "协作": ["cooperative", "collaborative"],
                    "分布式": ["distributed"],
                    "去中心化": ["decentralized"],
                    "博弈": ["game theory", "game"],
                }
                for key, vals in mapping.items():
                    if key in q:
                        extra_terms.extend(vals)
                if ("多智能体" in q and "强化学习" in q):
                    extra_terms.extend(['"multi-agent reinforcement learning"', 'MARL'])
                    rl_hint = True
            if "marl" in q_lower:
                rl_hint = True
                extra_terms.extend(['"multi-agent reinforcement learning"', 'MARL'])

            all_include = list(include_terms or []) + extra_terms

            parts = [f"({q})"]
            if all_include:
                inc = " OR ".join([f"ti:{t} OR abs:{t}" for t in all_include])
                parts.append(f"({inc})")
            cats = list(categories or [])
            if rl_hint and not cats:
                cats = ["cs.LG", "cs.AI", "cs.MA"]
            if cats:
                cat = " OR ".join([f"cat:{c}" for c in cats])
                parts.append(f"({cat})")

            merged_query = " AND ".join(parts)

            search = arxiv.Search(
                query=merged_query,
                max_results=max_results,
                sort_by=(arxiv.SortCriterion.SubmittedDate if sort_by_recent else arxiv.SortCriterion.Relevance)
            )

            results: List[Dict[str, Any]] = []
            max_retries = 3
            base_delay = 3.0
            for attempt in range(max_retries):
                try:
                    for paper in client.results(search):
                        title_lower = (paper.title or "").lower()
                        summary_lower = (paper.summary or "").lower()
                        if exclude_terms and any(k.lower() in title_lower or k.lower() in summary_lower for k in exclude_terms):
                            continue

                        # 日期窗口过滤（例如最近 180/365 天）
                        if days_window and paper.published:
                            try:
                                from datetime import datetime, timedelta
                                pub_date = paper.published.date()
                                if datetime.utcnow().date() - pub_date > timedelta(days=days_window):
                                    continue
                            except Exception:
                                pass

                        results.append({
                            "source": "arxiv",
                            "id": paper.get_short_id(),
                            "title": paper.title,
                            "authors": [a.name for a in paper.authors],
                            "summary": paper.summary,
                            "pdf_url": paper.pdf_url,
                            "published": str(paper.published.date()) if paper.published else None,
                            "url": paper.entry_id
                        })
                    break  # 成功
                except Exception as e:
                    msg = str(e).lower()
                    if any(tok in msg for tok in ["429", "rate limit", "proxy", "503", "remote end closed"]):
                        wait = base_delay * (attempt + 1) + random.uniform(0, 0.5)
                        print(f"[arXiv] 请求受限或代理异常，{wait:.1f}s 后重试 ({attempt+1}/{max_retries})")
                        await asyncio.sleep(wait)
                        continue
                    raise
            # 若无结果且触发 RL/MARL 语义，回退一次英文强检索
            if not results and rl_hint:
                fallback_terms = [
                    'ti:"multi-agent reinforcement learning" OR abs:"multi-agent reinforcement learning"',
                    'ti:MARL OR abs:MARL'
                ]
                fb_parts = ["(" + ") AND (".join(fallback_terms) + ")"]
                if cats:
                    fb_parts.append("(" + " OR ".join([f"cat:{c}" for c in cats]) + ")")
                fb_query = " AND ".join(fb_parts)

                search_fb = arxiv.Search(
                    query=fb_query,
                    max_results=max_results,
                    sort_by=(arxiv.SortCriterion.SubmittedDate if sort_by_recent else arxiv.SortCriterion.Relevance)
                )
                try:
                    for paper in client.results(search_fb):
                        results.append({
                            "source": "arxiv",
                            "id": paper.get_short_id(),
                            "title": paper.title,
                            "authors": [a.name for a in paper.authors],
                            "summary": paper.summary,
                            "pdf_url": paper.pdf_url,
                            "published": str(paper.published.date()) if paper.published else None,
                            "url": paper.entry_id
                        })
                except Exception as e:
                    print(f"[arXiv] 回退英文检索失败: {e}")

            return results
        except Exception as e:
            print(f"[Scanner] arXiv 搜索失败: {e}")
            return []
    
    async def search_google_scholar(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """搜索 Google Scholar（可选，增加超时与健壮性回退）"""
        if not self.use_scholar:
            return []
        try:
            import asyncio
            from scholarly import scholarly

            async def _do_search():
                results = []
                try:
                    search_query = scholarly.search_pubs(query)
                    count = 0
                    for pub in search_query:
                        if count >= max_results:
                            break
                        try:
                            pub_filled = scholarly.fill(pub)
                            results.append({
                                "source": "google_scholar",
                                "title": pub_filled.get("bib", {}).get("title", ""),
                                "authors": pub_filled.get("bib", {}).get("author", []),
                                "year": pub_filled.get("bib", {}).get("pub_year", ""),
                                "url": pub_filled.get("pub_url", ""),
                                "pdf_url": pub_filled.get("eprint_url", ""),
                                "citations": pub_filled.get("num_citations", 0)
                            })
                            count += 1
                        except Exception as e:
                            print(f"[Scanner] 获取 Google Scholar 详情失败: {e}")
                            continue
                except Exception as e:
                    print(f"[Scanner] Google Scholar 搜索内部失败: {e}")
                return results

            try:
                return await asyncio.wait_for(_do_search(), timeout=self.scholar_timeout)
            except asyncio.TimeoutError:
                print(f"[Scanner] Google Scholar 搜索超时（{self.scholar_timeout}s），已跳过并回退到其它来源")
                return []
        except Exception as e:
            print(f"[Scanner] Google Scholar 搜索失败: {e}")
            print("[Scanner] 提示: 如果 scholarly 库不可用，将跳过 Google Scholar 搜索")
            return []

    def _load_local_leaderboards(self) -> List[Dict[str, Any]]:
        """加载本地榜单缓存（papers/*/papers_info.json）"""
        results: List[Dict[str, Any]] = []
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "papers"))
            for root, dirs, files in os.walk(base_dir):
                for f in files:
                    if f == "papers_info.json":
                        path = os.path.join(root, f)
                        try:
                            # 兼容多种编码与结构（list 或 dict）
                            data = None
                            last_err: Optional[Exception] = None
                            for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"):
                                try:
                                    with open(path, "r", encoding=enc, errors=("strict" if enc != "latin-1" else "ignore")) as fp:
                                        data = json.load(fp)
                                    last_err = None
                                    break
                                except Exception as e:
                                    last_err = e
                                    data = None
                            if data is None and last_err is not None:
                                # 尝试二进制读取 + 忽略错误再解析
                                try:
                                    with open(path, "rb") as fb:
                                        raw = fb.read()
                                    text = raw.decode("utf-8", errors="ignore")
                                    data = json.loads(text)
                                except Exception:
                                    raise last_err

                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict):
                                        item["source"] = item.get("source", "leaderboard")
                                        results.append(item)
                            elif isinstance(data, dict):
                                # 一些缓存是 {id: {...}} 形式
                                for _, item in data.items():
                                    if isinstance(item, dict):
                                        item["source"] = item.get("source", "leaderboard")
                                        results.append(item)
                        except Exception as e:
                            print(f"[Scanner] 读取榜单缓存失败: {path} -> {e}")
        except Exception as e:
            print(f"[Scanner] 遍历榜单缓存失败: {e}")
        return results
    
    async def search(self, query: str, max_results_per_source: int = 10) -> Dict[str, Any]:
        """多源搜索（优先稳定来源，Scholar 为可选补充）"""
        print(f"[Scanner] 开始搜索: {query}")

        # 先加载本地榜单缓存
        leaderboard_results = self._load_local_leaderboards()

        # 并行搜索 arXiv 与（可选）Scholar
        # 默认开启按最近提交排序，并应用一年内时间窗
        arxiv_results = await self.search_arxiv(
            query,
            max_results=max_results_per_source,
            sort_by_recent=True,
            days_window=365
        )
        scholar_results = await self.search_google_scholar(query, max_results_per_source)

        # 合并结果（稳定来源优先）
        total_results = len(leaderboard_results) + len(arxiv_results) + len(scholar_results)
        all_results = {
            "query": query,
            "leaderboard_results": leaderboard_results,
            "arxiv_results": arxiv_results,
            "google_scholar_results": scholar_results,
            "total_results": total_results,
            "timestamp": datetime.utcnow().isoformat(),
            "notes": "使用稳定来源优先；Scholar 为可选并带超时保护"
        }

        print(f"[Scanner] 找到 {len(leaderboard_results)} 个榜单缓存，{len(arxiv_results)} 个 arXiv 结果，{len(scholar_results)} 个 Google Scholar 结果")
        return all_results


# Agent B: Extractor
class ExtractorAgent:
    """Agent B: 从 PDF 提取文本和简单表格（支持 Vision Model 增强）"""
    
    def __init__(self, use_vision: bool = False, vision_model: str = "gpt-4o"):
        """
        初始化 Extractor
        
        Args:
            use_vision: 是否使用 Vision Model 增强
            vision_model: Vision Model 名称
        """
        self.name = "extractor"
        self.paper_cache_dir = "papers/extracted"
        os.makedirs(self.paper_cache_dir, exist_ok=True)
        
        self.use_vision = use_vision
        self.vision_extractor = None
        
        if use_vision:
            try:
                from .vision_extractor import VisionExtractor
                self.vision_extractor = VisionExtractor(vision_model)
                print(f"[Extractor] Vision Model 已启用: {vision_model}")
            except ImportError as e:
                print(f"[Extractor] Vision Extractor 导入失败: {e}，使用基础模式")
                self.use_vision = False
    
    def download_pdf(self, pdf_url: str, paper_id: str) -> Optional[str]:
        """下载 PDF（增强：重试、UA 头、arXiv 链接规范化、超时）"""
        import re
        try:
            import requests
            # 规范化 arXiv 链接：如果是 abs 页面，转为 pdf 下载
            if pdf_url and "arxiv.org" in pdf_url and "/abs/" in pdf_url:
                pdf_url = re.sub(r"/abs/([\w\.-]+)", r"/pdf/\1.pdf", pdf_url)

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
            pdf_path = os.path.join(self.paper_cache_dir, f"{paper_id}.pdf")

            # 简单重试机制
            for attempt in range(3):
                try:
                    response = requests.get(pdf_url, headers=headers, timeout=30)
                    if response.status_code == 200 and response.content:
                        with open(pdf_path, "wb") as f:
                            f.write(response.content)
                        return pdf_path
                    else:
                        print(f"[Extractor] 下载失败，状态码 {response.status_code}，尝试 {attempt+1}/3")
                except Exception as e:
                    print(f"[Extractor] 下载异常（尝试 {attempt+1}/3）: {e}")
            # 尝试跟随重定向一次
            try:
                response = requests.get(pdf_url, headers=headers, timeout=30, allow_redirects=True)
                if response.status_code == 200 and response.content:
                    with open(pdf_path, "wb") as f:
                        f.write(response.content)
                    return pdf_path
            except Exception as e:
                print(f"[Extractor] 重定向下载失败: {e}")
        except Exception as e:
            print(f"[Extractor] 下载 PDF 失败 {pdf_url}: {e}")
        return None
    
    def extract_text(self, pdf_path: str) -> str:
        """提取 PDF 文本（使用 PyMuPDF）"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"[Extractor] 文本提取失败: {e}")
            # 降级到 pdfplumber
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                return text
            except Exception as e2:
                print(f"[Extractor] pdfplumber 也失败: {e2}")
                return ""
    
    def extract_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """提取简单表格（使用 pdfplumber）"""
        try:
            import pdfplumber
            tables = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table_num, table in enumerate(page_tables):
                        if table and len(table) > 1:  # 至少要有表头和数据行
                            tables.append({
                                "page": page_num + 1,
                                "table_index": table_num,
                                "data": table,
                                "rows": len(table),
                                "cols": len(table[0]) if table else 0
                            })
            return tables
        except Exception as e:
            print(f"[Extractor] 表格提取失败: {e}")
            return []
    
    def extract_metrics_from_text(self, text: str, context: str = "") -> List[Dict[str, Any]]:
        """从文本中提取指标（支持 Vision Model 增强；强化跟踪领域常用指标）"""
        if self.use_vision and self.vision_extractor:
            # 使用 LLM 进行上下文理解
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环正在运行，使用线程池
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            self.vision_extractor.extract_metrics_with_llm(text, context)
                        )
                        metrics = future.result()
                else:
                    metrics = loop.run_until_complete(
                        self.vision_extractor.extract_metrics_with_llm(text, context)
                    )
                
                if metrics:
                    return metrics
            except Exception as e:
                print(f"[Extractor] Vision 指标提取失败，降级到基础模式: {e}")
        
        # 降级到基础正则表达式提取
        import re
        metrics = []
        
        # 常见指标模式（含跟踪领域：AO、SR、AUC、Precision、Success）
        metric_patterns = [
            (r"(?:accuracy|acc)\s*[=:]\s*(\d+\.?\d*)\s*%?", "accuracy"),
            (r"(?:f1[- ]?score|f1)\s*[=:]\s*(\d+\.?\d*)\s*%?", "f1_score"),
            (r"(?:mAP|mean average precision)\s*[=:]\s*(\d+\.?\d*)\s*%?", "mAP"),
            (r"(?:top[- ]?1|top1)\s*[=:]\s*(\d+\.?\d*)\s*%?", "top1_accuracy"),
            (r"(?:top[- ]?5|top5)\s*[=:]\s*(\d+\.?\d*)\s*%?", "top5_accuracy"),
            # Tracking 常用
            (r"\bAO\b\s*[=:]\s*(\d+\.?\d*)\s*%?", "ao"),
            (r"\bSR\b\s*(?:@?0?\.?5)?\s*[=:]\s*(\d+\.?\d*)\s*%?", "sr"),
            (r"\bAUC\b\s*[=:]\s*(\d+\.?\d*)\s*%?", "auc"),
            (r"(?:precision|prec)\s*[=:]\s*(\d+\.?\d*)\s*%?", "precision"),
            (r"(?:success rate|success)\s*[=:]\s*(\d+\.?\d*)\s*%?", "success_rate"),
        ]
        
        for pattern, metric_name in metric_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = float(match.group(1))
                # 如果是 0-1 范围，转换为百分比
                if value <= 1.0:
                    value = value * 100
                
                metrics.append({
                    "metric": metric_name,
                    "value": value,
                    "unit": "percentage",
                    "context": text[max(0, match.start()-50):match.end()+50]
                })
        
        return metrics
    
    async def extract(self, paper_info: Dict[str, Any]) -> Dict[str, Any]:
        """提取论文信息"""
        paper_id = paper_info.get("id", "unknown")
        pdf_url = paper_info.get("pdf_url", "")
        published = paper_info.get("published")
        source = paper_info.get("source", "unknown")
        
        if not pdf_url:
            return {
                "paper_id": paper_id,
                "status": "no_pdf",
                "error": "没有 PDF URL"
            }
        
        print(f"[Extractor] 开始提取: {paper_id}")
        
        # 下载 PDF
        pdf_path = self.download_pdf(pdf_url, paper_id)
        if not pdf_path:
            return {
                "paper_id": paper_id,
                "status": "download_failed",
                "error": "PDF 下载失败"
            }
        
        # 提取文本
        text = self.extract_text(pdf_path)
        
        # 提取表格
        tables = self.extract_tables(pdf_path)
        
        # 如果启用 Vision Model，尝试增强表格提取
        if self.use_vision and self.vision_extractor:
            try:
                # 检测表格位置
                table_locations = self.vision_extractor.detect_tables_in_pdf(pdf_path)
                print(f"[Extractor] 检测到 {len(table_locations)} 个表格位置")
                
                # 对每个表格使用 Vision Model 精提取（可选）
                # 这里可以添加 Vision Model 处理逻辑
            except Exception as e:
                print(f"[Extractor] Vision 表格处理失败: {e}")
        
        # 从文本中提取指标（支持 Vision Model 增强）
        context = f"Title: {paper_info.get('title', '')}\nSummary: {paper_info.get('summary', '')[:500]}"
        metrics = self.extract_metrics_from_text(text, context)
        
        result = {
            "paper_id": paper_id,
            "title": paper_info.get("title", ""),
            "published": published,
            "source": source,
            "status": "success",
            "text_length": len(text),
            "tables_count": len(tables),
            "metrics_count": len(metrics),
            "metrics": metrics,
            "tables": tables[:5],  # 只保留前5个表格
            "text_preview": text[:1000]  # 文本预览
        }
        
        print(f"[Extractor] 提取完成: {len(metrics)} 个指标，{len(tables)} 个表格")
        return result


# Agent C: Normalizer
class NormalizerAgent:
    """Agent C: 指标标准化和转换"""
    
    def __init__(self):
        self.name = "normalizer"
        
        # 指标转换规则（扩展版）
        self.metric_conversions = {
            "error_rate": lambda x: 100 - x,  # Error Rate -> Accuracy
            "err": lambda x: 100 - x,
            "error": lambda x: 100 - x,
            "classification_error": lambda x: 100 - x,
            "misclassification_rate": lambda x: 100 - x,
            # 注意：F1 和 Accuracy 不能直接转换，需要上下文
        }
        
        # 数据集别名映射（扩展版）
        self.dataset_aliases = {
            "imagenet": ["ILSVRC", "ImageNet-1K", "ImageNet", "ImageNet-1k", "ImageNet1K", "ILSVRC2012"],
            "cifar-10": ["CIFAR-10", "CIFAR10", "cifar10", "CIFAR 10"],
            "cifar-100": ["CIFAR-100", "CIFAR100", "cifar100", "CIFAR 100"],
            "got-10k": ["GOT-10k", "GOT10k", "got10k", "GOT-10K"],
            "lasot": ["LaSOT", "LaSOT", "lasot"],
            "trackingnet": ["TrackingNet", "trackingnet", "Tracking Net"],
            "coco": ["COCO", "coco", "MS COCO", "mscoco"],
            "pascal_voc": ["PASCAL VOC", "Pascal VOC", "VOC", "voc"],
            "cityscapes": ["Cityscapes", "cityscapes", "CityScapes"],
            "otb": ["OTB", "OTB100", "OTB-100", "otb"],
            "uav123": ["UAV123", "uav123"],
            "nfs": ["NFS", "Need for Speed"],
            "tpl": ["TLP", "Tracking-Learning-Prediction"],
        }
        
        # 指标等价关系（用于标准化）
        self.metric_equivalences = {
            "accuracy": ["acc", "accuracy", "classification accuracy", "top-1 accuracy"],
            "top1_accuracy": ["top-1", "top1", "top 1", "top-1 accuracy", "top1 accuracy"],
            "top5_accuracy": ["top-5", "top5", "top 5", "top-5 accuracy", "top5 accuracy"],
            "f1_score": ["f1", "f1-score", "f1 score", "f1score", "f-measure"],
            "map": ["mAP", "mean average precision", "mean ap", "map"],
            "iou": ["IoU", "iou", "intersection over union", "jaccard index"],
            # Tracking 常见指标等价
            "ao": ["ao", "average overlap"],
            "sr": ["sr", "success rate", "success"],
            "auc": ["auc", "area under curve"],
            "precision": ["precision", "prec"],
            "success_rate": ["success rate", "success"],
        }
        
        # 指标标准化名称（使用等价关系）
        self.metric_standard_names = {}
        for standard_name, variants in self.metric_equivalences.items():
            self.metric_standard_names[standard_name] = variants
    
    def normalize_metric_name(self, metric_name: str) -> str:
        """标准化指标名称"""
        metric_lower = metric_name.lower().strip()
        for standard_name, variants in self.metric_standard_names.items():
            if metric_lower in variants:
                return standard_name
        return metric_name.lower()
    
    def normalize_dataset_name(self, dataset_name: str) -> str:
        """标准化数据集名称"""
        dataset_lower = dataset_name.lower().strip()
        for standard_name, aliases in self.dataset_aliases.items():
            if dataset_lower in aliases or dataset_lower == standard_name:
                return standard_name
        return dataset_name
    
    def normalize_value(self, value: float, unit: str) -> Tuple[float, str]:
        """标准化数值和单位"""
        # 统一转换为百分比
        if unit in ["decimal", "ratio", "fraction"]:
            return value * 100, "percentage"
        elif unit == "percentage":
            return value, "percentage"
        else:
            # 默认假设是百分比
            if 0 <= value <= 1:
                return value * 100, "percentage"
            return value, "percentage"
    
    def convert_metric(self, metric_name: str, value: float) -> Optional[float]:
        """转换指标（如 Error Rate -> Accuracy）"""
        metric_lower = metric_name.lower()
        if metric_lower in self.metric_conversions:
            return self.metric_conversions[metric_lower](value)
        return None
    
    async def normalize(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化提取的数据"""
        print(f"[Normalizer] 开始标准化: {extracted_data.get('paper_id', 'unknown')}")
        
        normalized_metrics = []
        
        for metric in extracted_data.get("metrics", []):
            metric_name = metric.get("metric", "")
            value = metric.get("value", 0)
            unit = metric.get("unit", "percentage")
            
            # 标准化指标名称
            normalized_name = self.normalize_metric_name(metric_name)
            
            # 标准化数值
            normalized_value, normalized_unit = self.normalize_value(value, unit)
            
            # 尝试转换（如 Error Rate -> Accuracy）
            converted_value = self.convert_metric(metric_name, normalized_value)
            if converted_value is not None:
                normalized_metrics.append({
                    "original_metric": metric_name,
                    "normalized_metric": "accuracy",  # Error Rate 转换为 Accuracy
                    "original_value": normalized_value,
                    "normalized_value": converted_value,
                    "unit": normalized_unit,
                    "converted": True,
                    "context": metric.get("context", "")
                })
            else:
                normalized_metrics.append({
                    "original_metric": metric_name,
                    "normalized_metric": normalized_name,
                    "original_value": value,
                    "normalized_value": normalized_value,
                    "unit": normalized_unit,
                    "converted": False,
                    "context": metric.get("context", "")
                })
        
        result = {
            "paper_id": extracted_data.get("paper_id", ""),
            "title": extracted_data.get("title", ""),
            "normalized_metrics": normalized_metrics,
            "metrics_count": len(normalized_metrics),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        print(f"[Normalizer] 标准化完成: {len(normalized_metrics)} 个指标")
        return result


# Agent D: Verifier
class VerifierAgent:
    """Agent D: 冲突检测和验证（增强版）"""
    
    def __init__(self, conflict_threshold: float = 1.0):
        """
        初始化 Verifier
        
        Args:
            conflict_threshold: 冲突阈值（百分比），差异超过此值视为冲突
        """
        self.name = "verifier"
        self.conflict_threshold = conflict_threshold
        
        # 来源可信度权重
        self.source_weights = {
            "arxiv": 1.0,  # arXiv 官方发布，可信度高
            "google_scholar": 0.8,  # Google Scholar 聚合，可信度中等
            "paper_pdf": 0.9,  # 直接从论文 PDF 提取，可信度高
            "web": 0.6,  # 网页来源，可信度较低
        }
        
        # 时间新鲜度权重（越新越可信）
        self.time_decay_factor = 0.1  # 每年衰减 10%
    
    def calculate_confidence_score(self, paper_info: Dict[str, Any], metric_info: Dict[str, Any]) -> float:
        """
        计算单个指标的置信度评分
        
        Args:
            paper_info: 论文信息
            metric_info: 指标信息
        
        Returns:
            置信度评分 (0-1)
        """
        score = 1.0
        
        # 1. 来源可信度
        source = paper_info.get("source", "unknown")
        source_weight = self.source_weights.get(source, 0.5)
        score *= source_weight
        
        # 2. 指标数量（指标越多，提取越可靠）
        metrics_count = len(paper_info.get("normalized_metrics", []))
        if metrics_count > 0:
            score *= min(1.0, 0.5 + metrics_count / 10.0)  # 最多 10 个指标达到满分
        
        # 3. 上下文完整性（有上下文说明提取更准确）
        context = metric_info.get("context", "")
        if len(context) > 50:
            score *= 1.1  # 有上下文加分
        score = min(1.0, score)  # 限制在 1.0
        
        # 4. 转换状态（如果经过转换，可能引入误差）
        if metric_info.get("converted", False):
            score *= 0.95  # 转换过的指标稍微降权
        
        return score
    
    def find_conflicts(self, normalized_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测冲突（增强版：包含置信度分析）"""
        print(f"[Verifier] 开始验证: {len(normalized_results)} 个结果")
        
        # 按指标和数据集分组
        metric_groups: Dict[str, List[Dict[str, Any]]] = {}
        
        for result in normalized_results:
            for metric in result.get("normalized_metrics", []):
                metric_name = metric.get("normalized_metric", "")
                value = metric.get("normalized_value", 0)
                
                key = metric_name
                if key not in metric_groups:
                    metric_groups[key] = []
                
                # 计算置信度
                confidence = self.calculate_confidence_score(result, metric)
                
                metric_groups[key].append({
                    "paper_id": result.get("paper_id", ""),
                    "title": result.get("title", ""),
                    "source": result.get("source", "unknown"),
                    "metric": metric_name,
                    "value": value,
                    "original_metric": metric.get("original_metric", ""),
                    "context": metric.get("context", ""),
                    "confidence": confidence
                })
        
        # 检测冲突（考虑置信度）
        conflicts = []
        for metric_name, values in metric_groups.items():
            if len(values) < 2:
                continue
            
            # 计算加权平均（按置信度加权）
            weighted_sum = sum(v["value"] * v["confidence"] for v in values)
            confidence_sum = sum(v["confidence"] for v in values)
            weighted_avg = weighted_sum / confidence_sum if confidence_sum > 0 else sum(v["value"] for v in values) / len(values)
            
            # 简单平均
            value_list = [v["value"] for v in values]
            avg_value = sum(value_list) / len(value_list)
            max_value = max(value_list)
            min_value = min(value_list)
            diff = max_value - min_value
            
            # 计算标准差
            variance = sum((v["value"] - avg_value) ** 2 for v in values) / len(values)
            std_dev = variance ** 0.5
            
            if diff > self.conflict_threshold:
                # 找出高置信度和低置信度的值
                high_conf_values = [v for v in values if v["confidence"] > 0.7]
                low_conf_values = [v for v in values if v["confidence"] < 0.5]
                
                conflicts.append({
                    "metric": metric_name,
                    "papers": values,
                    "avg_value": avg_value,
                    "weighted_avg": weighted_avg,
                    "max_value": max_value,
                    "min_value": min_value,
                    "difference": diff,
                    "std_dev": std_dev,
                    "high_confidence_count": len(high_conf_values),
                    "low_confidence_count": len(low_conf_values),
                    "conflict_level": "high" if diff > 5.0 else "medium" if diff > 2.0 else "low",
                    "recommendation": self._generate_recommendation(values, weighted_avg, diff)
                })
        
        print(f"[Verifier] 发现 {len(conflicts)} 个潜在冲突")
        return conflicts
    
    def _generate_recommendation(self, values: List[Dict[str, Any]], weighted_avg: float, diff: float) -> str:
        """生成冲突解决建议"""
        high_conf = [v for v in values if v.get("confidence", 0) > 0.7]
        
        if len(high_conf) > 0:
            # 如果有高置信度的值，推荐使用加权平均
            return f"建议使用加权平均值 {weighted_avg:.2f}%（基于置信度），差异 {diff:.2f}% 可能由于不同实验设置导致"
        else:
            # 如果没有高置信度的值，建议进一步验证
            return f"所有值的置信度都较低，建议检查原始论文或使用更多来源验证"
    
    async def verify(self, normalized_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证结果"""
        conflicts = self.find_conflicts(normalized_results)
        
        # 计算置信度评分（增强版）
        confidence_scores = []
        for result in normalized_results:
            # 为每个指标计算置信度
            metric_confidences = []
            for metric in result.get("normalized_metrics", []):
                conf = self.calculate_confidence_score(result, metric)
                metric_confidences.append(conf)
            
            # 论文整体置信度 = 指标置信度的平均值
            overall_confidence = sum(metric_confidences) / len(metric_confidences) if metric_confidences else 0.5
            
            confidence_scores.append({
                "paper_id": result.get("paper_id", ""),
                "title": result.get("title", ""),
                "source": result.get("source", "unknown"),
                "overall_confidence": overall_confidence,
                "metrics_count": len(metric_confidences),
                "metric_confidences": metric_confidences,
                "confidence_level": "high" if overall_confidence > 0.7 else "medium" if overall_confidence > 0.5 else "low"
            })
        
        result = {
            "total_papers": len(normalized_results),
            "conflicts": conflicts,
            "conflicts_count": len(conflicts),
            "confidence_scores": confidence_scores,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return result


# Pipeline 协调器
class SOTAPipeline:
    """Multi-Agent Pipeline 主协调器"""
    
    def __init__(self, use_vision: bool = False, vision_model: str = "gpt-4o", use_scholar: bool = False, scholar_timeout: float = 12.0):
        """
        初始化 Pipeline
        
        Args:
            use_vision: 是否使用 Vision Model 增强提取
            vision_model: Vision Model 名称
        """
        self.scanner = ScannerAgent(use_scholar=use_scholar, scholar_timeout=scholar_timeout)
        self.extractor = ExtractorAgent(use_vision=use_vision, vision_model=vision_model)
        self.normalizer = NormalizerAgent()
        self.verifier = VerifierAgent()
    
    async def run(self, query: str, max_papers: int = 5) -> Dict[str, Any]:
        """运行完整 Pipeline"""
        print(f"\n{'='*60}")
        print(f"🚀 启动 SOTA Pipeline: {query}")
        print(f"{'='*60}\n")
        
        # Step 1: Scanner - 搜索论文
        print("📚 Step 1: Scanner Agent - 搜索论文...")
        search_results = await self.scanner.search(query, max_results_per_source=max_papers)
        
        # 合并所有论文
        all_papers = []
        # 优先合并本地榜单与 arXiv，再补充 Scholar
        all_papers.extend(search_results.get("leaderboard_results", []))
        all_papers.extend(search_results.get("arxiv_results", []))
        all_papers.extend(search_results.get("google_scholar_results", []))
        
        if not all_papers:
            return {
                "status": "no_results",
                "query": query,
                "message": "没有找到相关论文"
            }
        
        # 限制处理数量
        papers_to_process = all_papers[:max_papers]
        print(f"📄 将处理 {len(papers_to_process)} 篇论文（来源优先：leaderboard/arXiv → Scholar）\n")
        
        # Step 2: Extractor - 提取信息
        print("🔍 Step 2: Extractor Agent - 提取 PDF 信息...")
        extracted_results = []
        for paper in papers_to_process:
            extracted = await self.extractor.extract(paper)
            if extracted.get("status") == "success":
                extracted_results.append(extracted)
        
        if not extracted_results:
            return {
                "status": "extraction_failed",
                "query": query,
                "message": "PDF 提取失败"
            }
        
        print(f"✅ 成功提取 {len(extracted_results)} 篇论文\n")
        
        # Step 3: Normalizer - 标准化
        print("📊 Step 3: Normalizer Agent - 标准化指标...")
        normalized_results = []
        for extracted in extracted_results:
            normalized = await self.normalizer.normalize(extracted)
            normalized_results.append(normalized)
        
        print(f"✅ 标准化完成\n")
        
        # Step 4: Verifier - 验证
        print("🔎 Step 4: Verifier Agent - 验证和冲突检测...")
        verification = await self.verifier.verify(normalized_results)
        
        print(f"✅ 验证完成\n")
        
        # 排序与去重：发布时间优先，其次来源可信度，其次主指标
        def src_weight(src: str) -> int:
            return {"leaderboard": 3, "arxiv": 2, "google_scholar": 1}.get(src, 0)

        def parse_date(s: Optional[str]) -> float:
            if not s:
                return 0.0
            try:
                return datetime.fromisoformat(s).timestamp()
            except Exception:
                return 0.0

        def metric_score(result: Dict[str, Any]) -> float:
            # 选取可能的主指标（优先 ao/sr/auc/map/accuracy），缺省 0
            metrics_map = {}
            for m in result.get("normalized_metrics", []):
                metrics_map[m.get("normalized_metric")] = m.get("normalized_value")
            for k in ["ao", "sr", "auc", "map", "accuracy"]:
                v = metrics_map.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
            return 0.0

        # 去重（按标题规范化）
        def norm(s: Optional[str]) -> str:
            return (s or "").strip().lower()

        seen_titles = set()
        deduped_normalized = []
        for r in normalized_results:
            t = norm(r.get("title"))
            if t in seen_titles:
                continue
            seen_titles.add(t)
            deduped_normalized.append(r)

        deduped_normalized.sort(
            key=lambda x: (
                parse_date(x.get("published")),
                src_weight(x.get("source", "")),
                metric_score(x)
            ),
            reverse=True
        )

        # 汇总结果
        final_result = {
            "status": "success",
            "query": query,
            "pipeline_stages": {
                "scanner": {
                    "total_found": len(all_papers),
                    "processed": len(papers_to_process)
                },
                "extractor": {
                    "successful": len(extracted_results),
                    "failed": len(papers_to_process) - len(extracted_results)
                },
                "normalizer": {
                    "normalized_papers": len(deduped_normalized)
                },
                "verifier": {
                    "conflicts_found": verification.get("conflicts_count", 0)
                }
            },
            "normalized_results": deduped_normalized,
            "verification": verification,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        print(f"{'='*60}")
        print(f"✅ Pipeline 完成!")
        print(f"   - 处理论文: {len(normalized_results)}")
        print(f"   - 发现冲突: {verification.get('conflicts_count', 0)}")
        print(f"{'='*60}\n")
        
        return final_result


# 便捷函数
async def run_sota_pipeline(query: str, max_papers: int = 5, use_vision: bool = False, vision_model: str = "gpt-4o", use_scholar: bool = False, scholar_timeout: float = 12.0) -> Dict[str, Any]:
    """
    运行 SOTA Pipeline 的便捷函数
    
    Args:
        query: 搜索查询
        max_papers: 最多处理的论文数量
        use_vision: 是否使用 Vision Model 增强
        vision_model: Vision Model 名称
    """
    pipeline = SOTAPipeline(use_vision=use_vision, vision_model=vision_model, use_scholar=use_scholar, scholar_timeout=scholar_timeout)
    return await pipeline.run(query, max_papers)

