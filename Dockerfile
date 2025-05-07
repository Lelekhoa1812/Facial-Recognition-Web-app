FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential g++ make libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# App directory in writable space
ENV APP_HOME=/data
WORKDIR ${APP_HOME}
COPY . .

# Ensure font cache writable
ENV MPLCONFIGDIR=/tmp

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7860

# Run the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1"]
