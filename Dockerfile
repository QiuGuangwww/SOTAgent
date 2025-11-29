FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.10-slim

WORKDIR /app

# 配置 pip 镜像源以加速下载
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 设置 Python 输出不缓冲，以便及时看到日志
ENV PYTHONUNBUFFERED=1

COPY . /app/

# 使用镜像源可用的稳定版本，避免 gradio 高版本缺失或 SSL 问题
RUN pip install --no-cache-dir "huggingface_hub<0.25.0" "gradio==4.44.1"
RUN pip install --no-cache-dir google-adk google-genai arxiv \
    googlesearch-python requests beautifulsoup4 litellm
RUN pip install --no-cache-dir -r My_First_Agent/requirements_pipeline.txt || true

EXPOSE 50001

# 确保容器内访问 localhost/127.0.0.1 不走代理，避免 Gradio 的本地可达性自检失败
ENV NO_PROXY=127.0.0.1,localhost,::1
ENV no_proxy=127.0.0.1,localhost,::1
# 默认不开启 share（可用环境变量 GRADIO_SHARE=true 覆盖）
ENV GRADIO_SHARE=false
# 可抑制 Gemini-LiteLLM 的提示（非功能性）
ENV ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS=true

CMD ["python", "app.py"]