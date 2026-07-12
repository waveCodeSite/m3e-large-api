# 使用 slim 基础镜像（更小体积）
FROM python:3.10-slim

WORKDIR /app

# 安装 CPU 版 PyTorch >= 2.6（先安装，利用 Docker 缓存）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY localembedding.py .


EXPOSE 6008

CMD ["uvicorn", "localembedding:app", "--host", "127.0.0.1", "--port", "6008"]