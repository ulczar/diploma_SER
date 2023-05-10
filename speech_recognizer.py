import whisper
import torch
from emotion_recognizer import SpeechEmotionRecognizer

class SpeechToText:
    def __init__(self, model_name="small", energy_threshold=500, record_timeout=2, phrase_timeout=3, default_microphone=None):
        self.model = model_name
        self.audio_model = self.load_model()
        self.energy_threshold = energy_threshold
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout
        self.default_microphone = default_microphone
        self.ser = SpeechEmotionRecognizer()
    
    def load_model(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            return whisper.load_model(self.model, device=device)
        except Exception as err:
            print(err)
            raise