import os
import sys
import gradio as gr

# 将当前目录加入路径，确保能导入 app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("正在加载 app.py ...")
import app

# --- 定义 Mock 函数 ---
def mock_deduct_photon(request: gr.Request, amount: int) -> tuple[bool, str]:
    """
    模拟扣费函数：直接返回成功，绕过 API 请求
    """
    print(f"[Debug模式] 已拦截光子扣费请求：需扣除 {amount} 光子 -> 自动放行 ✅")
    return True, "【本地调试】扣费已自动跳过"

# --- 动态替换 (Monkey Patch) ---
print("正在应用补丁：替换 deduct_photon 为 mock_deduct_photon ...")
app.deduct_photon = mock_deduct_photon

# --- 启动应用 ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("   🚀 SotaAgent 本地调试模式启动")
    print("   ⚠️  注意：此模式下不会产生实际光子消耗")
    print("="*50 + "\n")
    
    # 使用 app.py 中定义的界面对象启动
    app.iface.launch(
        server_name='0.0.0.0',
        server_port=50001,
        share=False,
        show_error=True,
        show_api=False
    )
