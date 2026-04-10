from TTS.api import TTS
from faster_whisper import WhisperModel


class VoiceUtils:
    def __init__(self, voice):
        self.voice = voice

    def speak(self, text):
        tts = TTS(
            model_name=self.voice,
            progress_bar=False,
            gpu=False,
        )

        tts.tts_to_file(
            text=text,
            file_path="out.wav",
        )

    def transcribe(self, file_path):
        model = WhisperModel(
            "base",
            device="cpu",  # "cuda" falls GPU
            compute_type="int8",
        )

        segments, info = model.transcribe(file_path)
        transcription = " ".join(segment.text for segment in segments)
        return transcription
