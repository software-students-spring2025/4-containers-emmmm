FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libasound2-dev \
    portaudio19-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ML client code
COPY audio_recorder.py emotion_analyzer.py main.py db.py ./

# Create directory for audio data
RUN mkdir -p /app/audio_data

# Command to run when container starts
CMD ["python", "main.py"]