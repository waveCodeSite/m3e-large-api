# 使用 slim 基础镜像（更小体积）
FROM python:3.10-slim

WORKDIR /app

# 安装 CPU 版 PyTorch（先安装，利用 Docker 缓存）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY localembedding.py .

# 设置 HuggingFace 镜像（国内加速）
ENV HF_ENDPOINT=https://hf-mirror.com

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:6008/v1/embeddings', data=b'{\"input\":[\"ping\"],\"model\":\"m3e-large\"}')" 2>/dev/null || exit 1

EXPOSE 6008

CMD ["uvicorn", "localembedding:app", "--host", "0.0.0.0", "--port", "6008"]