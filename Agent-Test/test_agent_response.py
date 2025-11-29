"""测试Agent响应格式"""
import asyncio
from My_First_Agent.agent import root_agent

async def test_agent():
    message = "hello"
    print(f"发送消息: {message}")
    
    chunks = []
    async for chunk in root_agent.run_async(message):
        chunks.append(chunk)
        print(f"\n收到chunk:")
        print(f"  类型: {type(chunk)}")
        print(f"  类型名: {type(chunk).__name__}")
        
        if hasattr(chunk, '__dict__'):
            print(f"  __dict__键: {list(chunk.__dict__.keys())[:15]}")
        
        # 尝试访问常见属性
        for attr in ['text', 'content', 'output', 'message', 'response', 'result']:
            if hasattr(chunk, attr):
                try:
                    val = getattr(chunk, attr)
                    print(f"  {attr}: {type(val).__name__} = {str(val)[:100] if val else None}")
                except Exception as e:
                    print(f"  {attr}: 访问错误 - {e}")
    
    print(f"\n总共收到 {len(chunks)} 个chunks")
    if chunks:
        final = chunks[-1]
        print(f"\n最终chunk类型: {type(final).__name__}")

if __name__ == "__main__":
    asyncio.run(test_agent())




