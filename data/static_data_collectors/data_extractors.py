import librosa
import essentia
import essentia.standard as es
class Extractors:
    def tempo_estimator(file_path):
        y, sr = librosa.load(file_path, duration=30)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        return tempo

    def key_extractor(file_path): 
        
        audio = es.MonoLoader(filename=file_path)()
        
        # Initialize KeyExtractor   
        key_extractor = es.KeyExtractor()
        key, scale, strength = key_extractor(audio)
            
        print(key, scale)
    key_extractor('/home/ciona/projects/RCOLM/data/raw_data/GTZAN/genres_original/blues/blues.00011.wav')
