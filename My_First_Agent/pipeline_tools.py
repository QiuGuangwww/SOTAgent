"""
将 Multi-Agent Pipeline 集成到 ADK Agent 系统的工具函数
"""

import json
import asyncio
from typing import Dict, Any

# 延迟导入，避免循环依赖
try:
    from .multi_agent_pipeline import SOTAPipeline
    PIPELINE_MODULE_AVAILABLE = True
except ImportError as e:
    PIPELINE_MODULE_AVAILABLE = False
    print(f"[Warning] Multi-Agent Pipeline 模块不可用: {e}")

# 全局 Pipeline 实例
_pipeline_instance = None

def get_pipeline(use_vision: bool = False, vision_model: str = "gpt-4o"):
    """获取 Pipeline 实例（单例模式，支持 Vision Model）"""
    if not PIPELINE_MODULE_AVAILABLE:
        raise ImportError("Multi-Agent Pipeline 模块不可用，请安装依赖")
    
    global _pipeline_instance
    # 如果参数改变，重新创建实例
    if _pipeline_instance is None or (
        hasattr(_pipeline_instance, '_use_vision') and 
        _pipeline_instance._use_vision != use_vision
    ):
        _pipeline_instance = SOTAPipeline(use_vision=use_vision, vision_model=vision_model)
        _pipeline_instance._use_vision = use_vision
        _pipeline_instance._vision_model = vision_model
    return _pipeline_instance

def run_trustworthy_sota_search(query: str, max_papers: int = 5, use_vision: bool = False, vision_model: str = "gpt-4o") -> str:
    """
    运行可信 SOTA 搜索 Pipeline（同步版本，供 ADK Agent 调用）
    
    Args:
        query: 搜索查询（例如 "GOT-10k tracking SOTA"）
        max_papers: 最多处理的论文数量（默认 5）
        use_vision: 是否使用 Vision Model 增强提取（默认 False）
        vision_model: Vision Model 名称，可选 "gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash-exp"（默认 "gpt-4o"）
    
    Returns:
        JSON 字符串，包含完整的 Pipeline 结果
    """
    if not PIPELINE_MODULE_AVAILABLE:
        error_result = {
            "status": "error",
            "query": query,
            "error": "Pipeline 模块不可用",
            "message": "请安装依赖: pip install -r requirements_pipeline.txt"
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)
    
    try:
        pipeline = get_pipeline(use_vision=use_vision, vision_model=vision_model)
        
        # 运行异步 Pipeline
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环已经在运行，使用线程池
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, pipeline.run(query, max_papers))
                result = future.result()
        else:
            result = loop.run_until_complete(pipeline.run(query, max_papers))
        
        # 格式化结果供 Agent 使用
        formatted_result = format_pipeline_result(result)
        
        # 添加 Vision Model 使用信息
        if use_vision:
            formatted_result["vision_model_used"] = vision_model
            formatted_result["enhanced_extraction"] = True
        
        return json.dumps(formatted_result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        error_result = {
            "status": "error",
            "query": query,
            "error": str(e),
            "message": f"Pipeline 执行失败: {e}"
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)

def format_pipeline_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """格式化 Pipeline 结果，使其更易读"""
    if result.get("status") != "success":
        return result
    
    # 提取关键信息
    formatted = {
        "status": "success",
        "query": result.get("query", ""),
        "summary": {
            "total_papers_processed": len(result.get("normalized_results", [])),
            "conflicts_found": result.get("verification", {}).get("conflicts_count", 0),
            "total_metrics_extracted": sum(
                len(paper.get("normalized_metrics", []))
                for paper in result.get("normalized_results", [])
            )
        },
        "papers": []
    }
    
    # 格式化每篇论文的信息
    for paper in result.get("normalized_results", []):
        paper_info = {
            "title": paper.get("title", ""),
            "paper_id": paper.get("paper_id", ""),
            "metrics": []
        }
        
        # 提取指标
        for metric in paper.get("normalized_metrics", []):
            paper_info["metrics"].append({
                "metric": metric.get("normalized_metric", ""),
                "value": metric.get("normalized_value", 0),
                "unit": metric.get("unit", "percentage"),
                "original_metric": metric.get("original_metric", ""),
                "converted": metric.get("converted", False)
            })
        
        formatted["papers"].append(paper_info)
    
    # 添加冲突信息
    if result.get("verification", {}).get("conflicts"):
        formatted["conflicts"] = []
        for conflict in result.get("verification", {}).get("conflicts", [])[:5]:  # 只显示前5个冲突
            formatted["conflicts"].append({
                "metric": conflict.get("metric", ""),
                "difference": conflict.get("difference", 0),
                "conflict_level": conflict.get("conflict_level", "unknown"),
                "papers_involved": len(conflict.get("papers", []))
            })
    
    return formatted

