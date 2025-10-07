import librosa
import essentia
import essentia.standard as es
class Extractors:
    @staticmethod
    def tempo_estimator(file_path):
        y, sr = librosa.load(file_path, duration=30)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        return tempo, sr

    @staticmethod 
    def key_extractor(file_path): 
        
        audio = es.MonoLoader(filename=file_path)()
        
        # Initialize KeyExtractor   
        key_extractor = es.KeyExtractor()
        key, scale, strength = key_extractor(audio)
            
        
        return key, scale
