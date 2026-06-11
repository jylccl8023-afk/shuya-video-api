FROM python:3.11-slim

# 安装 ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# 创建输出目录
RUN mkdir -p /app/output_videos

# 环境变量（部署时覆盖）
ENV OUTPUT_DIR=/app/output_videos
ENV STATIC_BASE_URL=http://localhost:8000/videos
# Railway 会自动注入 PORT 环境变量，默认回退 8000
ENV PORT=8000

EXPOSE 8000

# 兼容 Railway 动态端口和 VPS 固定端口
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
