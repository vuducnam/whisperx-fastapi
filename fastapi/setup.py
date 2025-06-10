from setuptools import setup, find_packages

setup(
    name="whisperx-fastapi",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic-settings",
        "streaming-form-data",
        "torch",
        "torchaudio",
        "whisperx",
    ],
) 