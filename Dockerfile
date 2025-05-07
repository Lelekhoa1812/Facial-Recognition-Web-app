FROM python:3.11-slim

# Install essential packages
RUN apt-get update && apt-get install -y \
    build-essential g++ make libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces: safe, writable directory
ENV APP_HOME=/data
WORKDIR ${APP_HOME}
COPY . .

# Resolve Fontconfig Warnings
ENV MPLCONFIGDIR=/tmp

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Optional: Fix matplotlib permission issue
ENV MPLCONFIGDIR=/tmp

EXPOSE 7860
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1"]
