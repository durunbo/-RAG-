# 使用与本地开发环境一致的 Python 主次版本
FROM python:3.14-slim

# Python 运行配置
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# 容器内的项目目录
WORKDIR /app

# 安装 PyTorch 等依赖运行时可能需要的 OpenMP 库
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖清单，便于 Docker 缓存依赖安装层
COPY requirements.txt .

# 安装 Python 依赖
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# 复制应用代码
COPY app/ ./app/
COPY src/ ./src/

# 微服务监听端口
EXPOSE 8000

# 启动 FastAPI 服务
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]