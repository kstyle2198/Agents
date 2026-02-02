import os
import sounddevice as sd
import soundfile as sf
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()

# 파일 경로
filename = os.path.dirname(os.getcwd()) + "/audio.wav"
filename = filename.replace("\\", "/")

# ------------------------------
# 1. 파일이 없으면 녹음 생성
# ------------------------------
if not os.path.exists(filename):
    print("녹음 파일이 없습니다. 5초 동안 녹음합니다...")
    
    duration = 5  # 녹음 시간 (초)
    sample_rate = 44100  # 샘플링 레이트
    
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # 녹음 완료 대기
    
    sf.write(filename, recording, sample_rate)
    print(f"녹음 완료: {filename}")

# ------------------------------
# 2. 파일 읽어서 Whisper 처리
# ------------------------------
with open(filename, "rb") as file:
    transcription = client.audio.transcriptions.create(
        file=(filename, file.read()),
        model="whisper-large-v3-turbo",
        temperature=0,
        response_format="verbose_json",
    )
    print("=== Transcription ===")
    print(transcription.text)
