# WhisperX FastAPI

FastAPI implementation for WhisperX speech recognition.

## Prerequisites

- Python 3.10
- CUDA-compatible GPU
- NVIDIA drivers
- cuBLAS 11.x
- cuDNN 8.x

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create new environment
conda create -n whisperx python=3.10
conda activate whisperx

# Install dependencies
cd fastapi
pip install -r requirements-fastapi-cuda.txt
pip install -e .
```

### Option 2: Using venv

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
cd fastapi
pip install -r requirements-fastapi-cuda.txt
pip install -e .
```

## Configuration

1. Create `.env.cuda` file in `fastapi/env/`:
```env
TMP_DIR=tmp
CORS_ORIGINS=*
WHISPER_MODEL=large-v2
DEVICE=cuda
COMPUTE_TYPE=float16
BATCH_SIZE=16
LANGUAGE_CODE=auto
HF_API_KEY=your_huggingface_token  # Required for diarization
```

## Running FastAPI

```bash
# Make sure you're in the fastapi directory
cd fastapi

# Run with correct Python path
PYTHONPATH=$PYTHONPATH:. uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, you can access:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### 1. Upload and Transcribe
```bash
POST /transcribe/
Content-Type: multipart/form-data
Body: file=@your_audio_file.mp3
```

### 2. Check Status
```bash
GET /transcribe/status/{task_id}
```

### 3. Get Result
```bash
GET /transcribe/result/{task_id}
```

## Docker Support

```bash
# Build image
docker build -t whisperx-fastapi -f dockerization/Dockerfile .

# Run container
docker run -d -p 8000:8000 --gpus all whisperx-fastapi
```

## Dependencies

- Python 3.10
- PyTorch 2.1.0
- TorchVision 0.16.0
- TorchAudio 2.1.0
- FastAPI
- WhisperX

Note: Current implementation only supports GPU. CPU support contributions are welcome.