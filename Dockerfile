FROM python:3.11-slim

# OS packages for OpenCV & face_recognition
RUN apt-get update && apt-get install -y \
    build-essential cmake libgl1 libglib2.0-0 \
    libsm6 libxrender1 libxext6 libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
CMD ["python", "app.py"]
