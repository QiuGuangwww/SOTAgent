"""
测试 Multi-Agent Pipeline MVP
"""

import asyncio
import json
from multi_agent_pipeline import run_sota_pipeline

async def test_pipeline():
    """测试 Pipeline"""
    # 测试查询
    query = "GOT-10k tracking SOTA"
    max_papers = 3  # MVP 版本只处理少量论文
    
    print("开始测试 Multi-Agent Pipeline MVP...\n")
    
    try:
        result = await run_sota_pipeline(query, max_papers)
        
        # 保存结果
        output_file = "pipeline_result.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {output_file}")
        
        # 打印摘要
        print("\n" + "="*60)
        print("结果摘要:")
        print("="*60)
        print(f"状态: {result.get('status', 'unknown')}")
        
        if result.get("status") == "success":
            stages = result.get("pipeline_stages", {})
            print(f"\n处理阶段:")
            print(f"  - Scanner: 找到 {stages.get('scanner', {}).get('total_found', 0)} 篇论文")
            print(f"  - Extractor: 成功提取 {stages.get('extractor', {}).get('successful', 0)} 篇")
            print(f"  - Normalizer: 标准化 {stages.get('normalizer', {}).get('normalized_papers', 0)} 篇")
            print(f"  - Verifier: 发现 {stages.get('verifier', {}).get('conflicts_found', 0)} 个冲突")
            
            # 显示提取的指标
            print(f"\n提取的指标:")
            for paper in result.get("normalized_results", [])[:3]:  # 只显示前3篇
                print(f"\n  📄 {paper.get('title', 'Unknown')[:60]}")
                for metric in paper.get("normalized_metrics", [])[:3]:  # 每篇只显示前3个指标
                    print(f"     - {metric.get('normalized_metric', 'unknown')}: {metric.get('normalized_value', 0):.2f}%")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pipeline())


