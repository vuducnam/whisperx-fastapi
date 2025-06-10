# WhisperX FastAPI

FastAPI implementation for WhisperX speech recognition with GPU support.

## Features

- Automatic Speech Recognition (ASR) using WhisperX
- Word-level alignment
- Speaker diarization (optional)
- Multiple language support
- Real-time transcription
- GPU acceleration
- RESTful API with Swagger documentation

## Prerequisites

- Python 3.10
- CUDA-compatible GPU
- NVIDIA drivers
- cuBLAS 11.x
- cuDNN 8.x
- Hugging Face account (for speaker diarization)

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create and activate conda environment
conda create -n whisperx python=3.10
conda activate whisperx

# Install CUDA dependencies
conda install -c conda-forge cudnn=8.9.2

# Install requirements
pip install -r requirements-fastapi-cuda.txt
```

### Option 2: Using venv

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements-fastapi-cuda.txt
```

## Configuration

1. Create `.env.cuda` file in the `env` directory:

```bash
mkdir -p env
touch env/.env.cuda
```

2. Add the following configuration:

```env
TMP_DIR=tmp
CORS_ORIGINS=*
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=*
CORS_ALLOW_HEADERS=*
WHISPER_MODEL=large-v2
DEVICE=cuda
COMPUTE_TYPE=float16
BATCH_SIZE=16
LANGUAGE_CODE=auto
HF_API_KEY=your_huggingface_token  # Optional, for speaker diarization
FILE_LOADING_CHUNK_SIZE_MB=1024
TASK_CLEANUP_DELAY_MIN=60
MAX_FILE_SIZE_MB=4096
MAX_REQUEST_BODY_SIZE_MB=5000
MODEL_LOADING_RETRIES=3
MODEL_LOADING_RETRY_DELAY=5
```

### Getting Hugging Face Token (Optional)

To enable speaker diarization:

1. Create a free account at https://huggingface.co
2. Go to Settings -> Access Tokens
3. Create a new token
4. Accept the terms of use for the diarization model at https://huggingface.co/pyannote/speaker-diarization
5. Add your token to the `HF_API_KEY` in `.env.cuda`

Note: The free token is sufficient for personal use. The diarization model only needs to be downloaded once and will be cached locally.

## Running FastAPI

### Local Development

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Docker Support

```bash
# Build Docker image
docker build -t whisperx-fastapi .

# Run Docker container
docker run -d --gpus all -p 8000:8000 whisperx-fastapi
```

## API Documentation

Once the server is running, you can access:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### API Endpoints

#### 1. Upload and Transcribe Audio
```http
POST /transcribe/
```
- Upload an audio file for transcription
- Returns a task ID for tracking

#### 2. Check Transcription Status
```http
GET /transcribe/status/{task_id}
```
- Check the status of a transcription task
- Returns current status and creation time

#### 3. Get Transcription Result
```http
GET /transcribe/result/{task_id}
```
- Get the transcription result
- Returns segments, language, and speaker information (if diarization is enabled)

## Response Formats

### Transcription Response
```json
{
    "task_id": "string",
    "status": "string"
}
```

### Transcription Status
```json
{
    "task_id": "string",
    "status": "string",
    "creation_time": "datetime",
    "result": "object"
}
```

### Transcription Result
```json
{
    "segments": [
        {
            "start": "float",
            "end": "float",
            "text": "string",
            "speaker": "string"  // Only available if diarization is enabled
        }
    ],
    "language": "string",
    "speakers": ["string"]  // Only available if diarization is enabled
}
```

## Dependencies

- Python 3.10
- PyTorch 2.1.0
- TorchVision 0.16.0
- TorchAudio 2.1.0
- FastAPI 0.109.2
- WhisperX 3.3.0
- Pyannote.audio 3.1.1
- CUDA 11.x
- cuDNN 8.9.2

## Notes

- Current implementation only supports GPU
- CPU support is planned for future releases
- Speaker diarization is optional and requires a free Hugging Face token
- Maximum file size is configurable (default: 4GB)
- Tasks are automatically cleaned up after 60 minutes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 