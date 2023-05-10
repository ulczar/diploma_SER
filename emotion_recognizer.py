import torch
from transformers import Wav2Vec2FeatureExtractor
import librosa
import noisereduce as nr

class SpeechEmotionRecognizer:
    def __init__(self):
        self.model_file = 'models/ser_full_789.pkl'
        self.model = self.get_model()
        self.device = self.get_device()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")
        self.labels_map_inverse = {
            0: "angry",
            1: "positive",
            2: "neutral",
            3: "sad",
        }
        
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device

    @staticmethod   
    def clean_audio(data):
        data = nr.reduce_noise(data, sr=16000)
        xt, _ = librosa.effects.trim(data, top_db=33)
        return xt
     
    def get_model(self):
        return torch.load(self.model_file)
    
    def predict(self, audio_array):
        input_encodings = self.feature_extractor([self.clean_audio(audio_array)], sampling_rate=16000, padding=True, return_tensors="pt")

        input_values = input_encodings['input_values'].to(self.device)
        attention_mask = input_encodings['attention_mask'].to(self.device)

        with torch.no_grad():
            self.model.eval()
            output = self.model(input_values, attention_mask=attention_mask)
            

        predicted_class = torch.argmax(output.logits, dim=1).item()
        
        return self.labels_map_inverse[predicted_class]
