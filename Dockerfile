FROM python:3.11-slim

# OS packages for face_recognition, OpenCV, dlib
RUN apt-get update && apt-get install -y \
    build-essential cmake libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    libboost-all-dev libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Render will use
EXPOSE 7860

# Start via Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1"]
