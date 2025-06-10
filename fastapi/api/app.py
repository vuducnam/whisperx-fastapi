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
import subprocess
import numpy as np
import torchaudio


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
    supported_audio_formats: set = {'wav', 'mp3', 'm4a', 'ogg', 'flac'}
    supported_sampling_rates: set = {8000, 16000, 22050, 24000, 32000, 44100, 48000}
    default_sampling_rate: int = 16000
    default_channels: int = 1
    default_audio_format: str = 'wav'

    class Config:
        env_file = 'env/.env.cuda'
        env_file_encoding = 'utf-8'


@dataclass
class WhisperXModels:
    whisper_model: Any = None
    diarize_pipeline: Any = None
    align_model: Any = None
    align_model_metadata: Any = None

    def is_ready(self) -> bool:
        return self.whisper_model is not None

    def load_models(self, settings: 'TranscriptionAPISettings') -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load Whisper model
        self.whisper_model = whisperx.load_model(
            settings.whisper_model,
            settings.device,
            compute_type=settings.compute_type,
            language=settings.language_code if settings.language_code != "auto" else None
        )

        # Load alignment model if language is specified
        if settings.language_code != "auto":
            self.align_model, self.align_model_metadata = whisperx.load_align_model(
                language_code=settings.language_code,
                device=settings.device
            )

        # Load diarization pipeline if HF token is provided
        if settings.hf_api_key:
            self.diarize_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=settings.hf_api_key
            ).to(torch.device(settings.device))


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

whisperx_models = WhisperXModels()


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


def convert_audio(input_path: str, output_path: str, settings: TranscriptionAPISettings) -> None:
    try:
        result = subprocess.run([
            'ffmpeg', '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', str(settings.default_sampling_rate),
            '-ac', str(settings.default_channels),
            output_path
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert audio: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Error converting audio: {str(e)}")


def load_audio_file(file_path: str, settings: TranscriptionAPISettings) -> np.ndarray:
    try:
        audio = whisperx.load_audio(file_path)
        if audio is None or len(audio) == 0:
            raise RuntimeError("Failed to load audio: Empty audio data")
        return audio
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {str(e)}")


def transcribe_audio(audio_file_path: str, settings: TranscriptionAPISettings) -> dict:
    global whisperx_models

    if not whisperx_models.is_ready():
        raise RuntimeError("WhisperX models are not loaded")

    try:
        # Convert audio if needed
        if not audio_file_path.lower().endswith(settings.default_audio_format):
            wav_path = f"{audio_file_path.rsplit('.', 1)[0]}.{settings.default_audio_format}"
            convert_audio(audio_file_path, wav_path, settings)
            audio_file_path = wav_path

        # Load audio
        audio = load_audio_file(audio_file_path, settings)

        # Transcribe with Whisper
        transcription_result = whisperx_models.whisper_model.transcribe(
            audio,
            batch_size=settings.batch_size,
        )

        if not transcription_result or "segments" not in transcription_result:
            raise RuntimeError("Invalid transcription result: Missing segments")

        # Auto-detect language if not specified
        if settings.language_code == "auto":
            language = transcription_result["language"]
            whisperx_models.align_model, whisperx_models.align_model_metadata = whisperx.load_align_model(
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
            try:
                # Ensure file exists and is readable
                if not os.path.exists(audio_file_path):
                    raise RuntimeError(f"Audio file not found: {audio_file_path}")
                
                if not os.access(audio_file_path, os.R_OK):
                    raise RuntimeError(f"Audio file not readable: {audio_file_path}")

                # Get file info
                file_size = os.path.getsize(audio_file_path)
                if file_size == 0:
                    raise RuntimeError(f"Audio file is empty: {audio_file_path}")

                # Try to load audio with torchaudio first to validate
                try:
                    waveform, sample_rate = torchaudio.load(audio_file_path)
                    if waveform.numel() == 0:
                        raise RuntimeError(f"Invalid audio data in file: {audio_file_path}")
                except Exception as e:
                    raise RuntimeError(f"Failed to load audio with torchaudio: {str(e)}")

                # Perform diarization with error handling
                try:
                    # Convert audio to mono if needed
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                        torchaudio.save(audio_file_path, waveform, sample_rate)

                    # Run diarization
                    diarize_result = whisperx_models.diarize_pipeline(
                        audio_file_path,
                        min_speakers=1,
                        max_speakers=10
                    )

                    if not diarize_result:
                        raise RuntimeError("Diarization returned empty result")

                    # Check if result is already in the correct format
                    if isinstance(diarize_result, dict) and "segments" in diarize_result:
                        # Result is already in the correct format
                        pass
                    else:
                        # Convert diarization result to the format expected by whisperx
                        diarize_segments = []
                        for segment, track, speaker in diarize_result.itertracks(yield_label=True):
                            diarize_segments.append({
                                "start": float(segment.start),
                                "end": float(segment.end),
                                "speaker": f"SPEAKER_{speaker}"
                            })

                        if not diarize_segments:
                            raise RuntimeError("No valid segments found in diarization result")

                        diarize_result = {"segments": diarize_segments}

                    try:
                        # Ensure aligned_result has the correct format
                        if not isinstance(aligned_result, dict) or "segments" not in aligned_result:
                            raise RuntimeError("Invalid aligned result format")

                        # Ensure diarize_result has the correct format
                        if not isinstance(diarize_result, dict) or "segments" not in diarize_result:
                            raise RuntimeError("Invalid diarization result format")

                        # Assign speakers to words
                        final_result = whisperx.assign_word_speakers(
                            diarize_result,
                            aligned_result
                        )

                        # Extract unique speakers
                        speakers = list(set(segment.get("speaker", "") for segment in final_result["segments"] if segment.get("speaker")))
                        final_result["speakers"] = speakers

                    except Exception as e:
                        print(f"Error in assign_word_speakers: {str(e)}")
                        print(f"Aligned result type: {type(aligned_result)}")
                        print(f"Diarization result type: {type(diarize_result)}")
                        # Fallback to aligned result without speaker assignment
                        final_result = aligned_result
                        final_result["speakers"] = []

                except Exception as e:
                    print(f"Error in diarization pipeline: {str(e)}")
                    print(f"Diarization result type: {type(diarize_result)}")
                    if isinstance(diarize_result, (list, dict)):
                        print(f"Diarization result: {diarize_result}")
                    raise RuntimeError(f"Diarization pipeline error: {str(e)}")

            except Exception as e:
                print(f"Error in diarization: {str(e)}")
                print(f"Audio file path: {audio_file_path}")
                print(f"File exists: {os.path.exists(audio_file_path)}")
                if os.path.exists(audio_file_path):
                    print(f"File size: {os.path.getsize(audio_file_path)} bytes")
                final_result = aligned_result
                final_result["speakers"] = []
        else:
            final_result = aligned_result
            final_result["speakers"] = []

        final_result["language"] = transcription_result["language"]

        # Clean up temporary file after all processing is done
        if audio_file_path.endswith(settings.default_audio_format) and os.path.exists(audio_file_path):
            try:
                os.remove(audio_file_path)
            except Exception as e:
                print(f"Error removing temporary file: {str(e)}")

        return final_result

    except Exception as e:
        # Clean up temporary file in case of error
        if audio_file_path.endswith(settings.default_audio_format) and os.path.exists(audio_file_path):
            try:
                os.remove(audio_file_path)
            except Exception as e:
                print(f"Error removing temporary file: {str(e)}")
        print(f"Error in transcribe_audio: {str(e)}")
        raise RuntimeError(f"Transcription failed: {str(e)}")


def transcription_worker() -> None:
    while True:
        task_id, tmp_path = trancription_tasks_queue.get()

        try:
            # Update status to processing
            trancription_tasks[task_id].update({
                "status": "processing"
            })

            # Process audio
            result = transcribe_audio(tmp_path, settings)
            
            # Update task with success
            trancription_tasks[task_id].update({
                "status": "completed",
                "result": result
            })

        except Exception as e:
            print(f"Error processing task {task_id}: {str(e)}")
            # Update task with failure but don't raise error
            trancription_tasks[task_id].update({
                "status": "failed",
                "result": {
                    "segments": [],
                    "language": "unknown",
                    "speakers": []
                }
            })

        finally:
            trancription_tasks_queue.task_done()
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    print(f"Error removing temporary file {tmp_path}: {str(e)}")


def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("FFmpeg is not installed or not working properly")
        print("FFmpeg is installed and working properly")
        version_line = result.stdout.split('\n')[0]
        print(f"FFmpeg version: {version_line}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg is not installed. Please install FFmpeg first")
    except Exception as e:
        raise RuntimeError(f"Error checking FFmpeg: {str(e)}")


@app.on_event("startup")
async def startup_event():
    global whisperx_models
    try:
        # Check FFmpeg first
        check_ffmpeg()
        
        # Initialize models
        whisperx_models = WhisperXModels()
        whisperx_models.load_models(settings)
            
        print("All models loaded successfully")
        
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise

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
        "result": task.get("result", {
            "segments": [],
            "language": "unknown",
            "speakers": []
        })
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
    
    # If task is still processing, return current status
    if task["status"] in ["loading", "processing"]:
        return {
            "segments": [],
            "language": "unknown",
            "speakers": []
        }
    
    # If task failed, return empty result
    if task["status"] == "failed":
        return {
            "segments": [],
            "language": "unknown",
            "speakers": []
        }
    
    # If task completed, return result
    if task["status"] == "completed" and "result" in task:
        return task["result"]
    
    # Default case: return empty result
    return {
        "segments": [],
        "language": "unknown",
        "speakers": []
    }
