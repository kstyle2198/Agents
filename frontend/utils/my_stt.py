import os
import sounddevice as sd
import soundfile as sf
from groq import Groq
from dotenv import load_dotenv

from utils.setlogger import setup_logger
logger = setup_logger(f"{__name__}")

class AudioTranscriber:
    def __init__(self, filename=None, duration=5, sample_rate=44100, model="whisper-large-v3-turbo"):
        load_dotenv()
        self.client = Groq()
        self.duration = duration
        self.sample_rate = sample_rate
        self.model = model
        
        # 파일 경로 설정
        if filename is None:
            self.filename = os.path.join(os.path.dirname(os.getcwd()), "audio.wav").replace("\\", "/")
        else:
            self.filename = filename.replace("\\", "/")
    
    def record_audio(self):
        """파일이 없으면 녹음 생성"""
        if not os.path.exists(self.filename):
            logger.warning(f"녹음 파일이 없습니다. {self.duration}초 동안 녹음합니다...")
            recording = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1)
            sd.wait()
            sf.write(self.filename, recording, self.sample_rate)
            logger.info(f"녹음 완료: {self.filename}")
        else:
            logger.warning(f"이미 녹음 파일이 존재합니다: {self.filename}")
    
    def transcribe_audio(self):
        """녹음 파일을 읽어서 Whisper 모델로 변환"""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"{self.filename} 파일이 존재하지 않습니다.")
        
        with open(self.filename, "rb") as file:
            transcription = self.client.audio.transcriptions.create(
                file=(self.filename, file.read()),
                model=self.model,
                temperature=0,
                response_format="verbose_json",
            )
        
        # transcription 완료 후 파일 삭제
        try:
            os.remove(self.filename)
            logger.info(f"{self.filename} 파일을 삭제했습니다.")
        except Exception as e:
            logger.error(f"파일 삭제 중 오류 발생: {e}")

        return transcription.text
    
    def run(self):
        """전체 실행"""
        self.record_audio()
        text = self.transcribe_audio()
        logger.info("=== Transcription ===")
        logger.info(text)
        return text


if __name__ == "__main__":
    transcriber = AudioTranscriber()
    transcriber.run()
