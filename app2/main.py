# main.py

from fastapi import FastAPI
import logging
from dotenv import load_dotenv
import os
import shutil
import subprocess

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add FFmpeg path to PATH
ffmpeg_path = r"D:\Programmi\ffmpeg-master-latest-win64-gpl\bin"  # Update this path as needed
if not os.path.exists(ffmpeg_path):
    raise RuntimeError(f"FFmpeg path does not exist: {ffmpeg_path}")

if ffmpeg_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + ffmpeg_path

# Verify if FFmpeg is found
ffmpeg_executable = shutil.which("ffmpeg")
logger.info(f"FFmpeg executable path: {ffmpeg_executable}")

if ffmpeg_executable is None:
    raise RuntimeError("FFmpeg not found. Ensure the path is correct and that ffmpeg.exe exists in that directory.")
else:
    logger.info("FFmpeg found successfully.")

# Message for torchaudio
logger.info("Torchaudio is ready for use.")

# Initialize FastAPI
app = FastAPI(
    title="Sentiment & Emotion Analysis API",
    description="API for analyzing sentiment and emotions in tweets.",
    version="1.0.0"
)

# Include routers
from routers import (
    analyze_text,
    analyze_image,
    analyze_audio,
    analyze_multimodal,
    twitter
)

app.include_router(analyze_text.router)
app.include_router(analyze_image.router)
app.include_router(analyze_audio.router)
app.include_router(analyze_multimodal.router)
app.include_router(twitter.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment & Emotion Analysis API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")  # Set log_level to debug here as well
