import asyncio
import os
import uuid
import time
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
from threading import Thread
from typing import Any, Optional

from fastapi import HTTPException, BackgroundTasks, FastAPI, status, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from streaming_form_data import StreamingFormDataParser
from streaming_form_data.targets import FileTarget
from streaming_form_data.validators import MaxSizeValidator

import whisperx
import torch
from pyannote.audio import Pipeline


class TranscriptionResponse(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the transcription task")
    status: str = Field(..., description="Current status of the task (loading/processing/completed/failed)")


class TranscriptionStatus(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the transcription task")
    status: str = Field(..., description="Current status of the task")
    creation_time: datetime = Field(..., description="Time when the task was created")
    result: Optional[dict] = Field(None, description="Transcription result (available only when status is completed)")


class TranscriptionResult(BaseModel):
    segments: list = Field(..., description="List of transcribed segments")
    language: str = Field(..., description="Detected language code")
    speakers: Optional[list] = Field(None, description="List of detected speakers (if diarization is enabled)")


@dataclass
class WhisperXModels:
    whisper_model: Any
    diarize_pipeline: Any
    align_model: Any
    align_model_metadata: Any


class TranscriptionAPISettings(BaseSettings):
    tmp_dir: str = 'tmp'
    cors_origins: str = '*'
    cors_allow_credentials: bool = True
    cors_allow_methods: str = '*'
    cors_allow_headers: str = '*'
    whisper_model: str = 'large-v2'
    device: str = 'cuda'
    compute_type: str = 'float16'
    batch_size: int = 16
    language_code: str = 'auto'
    hf_api_key: str = ''
    file_loading_chunk_size_mb: int = 1024
    task_cleanup_delay_min: int = 60
    max_file_size_mb: int = 4096
    max_request_body_size_mb: int = 5000
    model_loading_retries: int = 3
    model_loading_retry_delay: int = 5

    class Config:
        env_file = 'env/.env.cuda'
        env_file_encoding = 'utf-8'


class MaxBodySizeException(Exception):
    def __init__(self, body_len: int):
        self.body_len = body_len


class MaxBodySizeValidator:
    def __init__(self, max_size: int):
        self.body_len = 0
        self.max_size = max_size

    def __call__(self, chunk: bytes):
        self.body_len += len(chunk)
        if self.body_len > self.max_size:
            raise MaxBodySizeException(self.body_len)


settings = TranscriptionAPISettings()

app = FastAPI(
    title="WhisperX FastAPI",
    description="""
    FastAPI implementation for WhisperX speech recognition.
    
    Features:
    - Automatic Speech Recognition (ASR)
    - Word-level alignment
    - Speaker diarization (optional)
    - Multiple language support
    - Real-time transcription
    """,
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(','),
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods.split(','),
    allow_headers=settings.cors_allow_headers.split(','),
)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="WhisperX API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    )


@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return get_openapi(
        title="WhisperX FastAPI",
        version="1.0.0",
        description="FastAPI implementation for WhisperX speech recognition",
        routes=app.routes,
    )


trancription_tasks = {}
trancription_tasks_queue = Queue()

whisperx_models = WhisperXModels(
    whisper_model=None,
    diarize_pipeline=None,
    align_model=None,
    align_model_metadata=None
)


def load_whisperx_models() -> None:
    global whisperx_models
    
    for attempt in range(settings.model_loading_retries):
        try:
            print(f"Loading WhisperX models (attempt {attempt + 1}/{settings.model_loading_retries})...")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Enable TF32 for better performance
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Load Whisper model
            whisperx_models.whisper_model = whisperx.load_model(
                whisper_arch=settings.whisper_model,
                device=settings.device,
                compute_type=settings.compute_type,
                language=settings.language_code if settings.language_code != "auto" else None
            )

            # Load diarization pipeline
            if settings.hf_api_key:
                whisperx_models.diarize_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=settings.hf_api_key
                ).to(torch.device(settings.device))

            # Load alignment model
            if settings.language_code != "auto":
                (
                    whisperx_models.align_model,
                    whisperx_models.align_model_metadata
                ) = whisperx.load_align_model(
                    language_code=settings.language_code,
                    device=settings.device
                )
            
            print("Successfully loaded all WhisperX models")
            return
            
        except Exception as e:
            print(f"Error loading models (attempt {attempt + 1}): {str(e)}")
            if attempt < settings.model_loading_retries - 1:
                print(f"Retrying in {settings.model_loading_retry_delay} seconds...")
                time.sleep(settings.model_loading_retry_delay)
            else:
                raise RuntimeError(f"Failed to load WhisperX models after {settings.model_loading_retries} attempts: {str(e)}")


def transcribe_audio(audio_file_path: str) -> dict:
    global whisperx_models

    # Load audio
    audio = whisperx.load_audio(audio_file_path)

    # Transcribe with Whisper
    transcription_result = whisperx_models.whisper_model.transcribe(
        audio,
        batch_size=int(settings.batch_size),
    )

    # Auto-detect language if not specified
    if settings.language_code == "auto":
        language = transcription_result["language"]
        (
            whisperx_models.align_model,
            whisperx_models.align_model_metadata
        ) = whisperx.load_align_model(
            language_code=language,
            device=settings.device
        )

    # Align whisper output
    aligned_result = whisperx.align(
        transcription_result["segments"],
        whisperx_models.align_model,
        whisperx_models.align_model_metadata,
        audio,
        settings.device,
        return_char_alignments=False
    )

    # Diarize if HF token is provided
    if settings.hf_api_key and whisperx_models.diarize_pipeline:
        diarize_segments = whisperx_models.diarize_pipeline(audio_file_path)
        final_result = whisperx.assign_word_speakers(
            diarize_segments,
            aligned_result
        )
        # Extract unique speakers
        speakers = list(set(segment.get("speaker", "") for segment in final_result["segments"] if segment.get("speaker")))
        final_result["speakers"] = speakers
    else:
        final_result = aligned_result
        final_result["speakers"] = []

    # Add language to final result
    final_result["language"] = transcription_result["language"]
    
    return final_result


def transcription_worker() -> None:
    while True:
        task_id, tmp_path = trancription_tasks_queue.get()

        try:
            result = transcribe_audio(tmp_path)
            trancription_tasks[task_id].update({"status": "completed", "result": result})

        except Exception as e:
            trancription_tasks[task_id].update({"status": "failed", "result": str(e)})

        finally:
            trancription_tasks_queue.task_done()
            os.remove(tmp_path)


@app.on_event("startup")
async def startup_event() -> None:
    os.makedirs(settings.tmp_dir, exist_ok=True)
    load_whisperx_models()
    Thread(target=transcription_worker, daemon=True).start()

    # Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


async def cleanup_task(task_id: str) -> None:
    await asyncio.sleep(settings.task_cleanup_delay_min * 60)
    trancription_tasks.pop(task_id, None)


@app.post("/transcribe/", response_model=TranscriptionResponse, tags=["Transcription"])
async def create_upload_file(
        file: UploadFile = File(..., description="Audio file to transcribe (supported formats: wav, mp3, m4a, etc.)"),
        background_tasks: BackgroundTasks = None
) -> dict:
    """
    Upload an audio file for transcription.
    
    - **file**: Audio file to transcribe (supported formats: wav, mp3, m4a, etc.)
    - Returns a task ID that can be used to check the status and get results
    """
    task_id = str(uuid.uuid4())
    tmp_path = f"{settings.tmp_dir}/{task_id}.audio"

    trancription_tasks[task_id] = {
        "status": "loading",
        "creation_time": datetime.utcnow(),
        "result": None
    }

    try:
        # Save uploaded file
        with open(tmp_path, "wb") as f:
            content = await file.read()
            if len(content) > settings.max_file_size_mb * 1024 * 1024:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File size exceeds maximum limit of {settings.max_file_size_mb}MB"
                )
            f.write(content)

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing upload: {str(e)}"
        )

    trancription_tasks_queue.put((task_id, tmp_path))
    background_tasks.add_task(cleanup_task, task_id)

    return {
        "task_id": task_id,
        "status": "loading"
    }


@app.get("/transcribe/status/{task_id}", response_model=TranscriptionStatus, tags=["Transcription"])
async def get_task_status(task_id: str) -> dict:
    """
    Get the status of a transcription task.
    
    - **task_id**: ID of the transcription task
    - Returns the current status and creation time
    """
    if task_id not in trancription_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    task = trancription_tasks[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "creation_time": task["creation_time"],
        "result": task["result"]
    }


@app.get("/transcribe/result/{task_id}", response_model=TranscriptionResult, tags=["Transcription"])
async def get_task_result(task_id: str) -> dict:
    """
    Get the result of a completed transcription task.
    
    - **task_id**: ID of the transcription task
    - Returns the transcription result with segments and speaker information
    """
    if task_id not in trancription_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    task = trancription_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task is not completed. Current status: {task['status']}"
        )

    if not task["result"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Task completed but no result available"
        )

    result = task["result"]
    
    return {
        "segments": result.get("segments", []),
        "language": result.get("language", "unknown"),
        "speakers": result.get("speakers", [])
    }
