# %% 
import librosa
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T 
import torchaudio.functional as F
import random
from librosa.effects import pitch_shift as lr_pitch_shift
# %%
def wav_to_mel_with_augmentation(filename, sr, duration, mel_save_path):
    try:
        db_snr = 7
        waveform, sr = torchaudio.load(filename)
        save_mel(waveform, sr, 1, mel_save_path, db_snr) 
        save_mel(waveform, sr, 2, mel_save_path, db_snr)
        save_mel(waveform, sr, 3, mel_save_path, db_snr)
        save_mel(waveform, sr, 4, mel_save_path, db_snr)


    except Exception as e:
        print(f'Błąd formatu pliku {filename}{e}')

#save melspec to png file 


def save_mel(waveform , sr, mode, mel_save_path, db_snr):
#mode 1 - no augmenttation 
#mode 2 - noise 
#mode 3 pitch schift
#mode 4 time stretch
#mode 5 SpecAugment 
    if mode == 1:

        waveform_np = waveform.squeeze().cpu().numpy()
        mel_spec = librosa.feature.melspectrogram(y=waveform_np, sr=sr, n_fft=2048, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_db)
        plt.savefig(mel_save_path, format='png')
        plt.close('all')
    
    if mode == 2:
        basename, ext = os.path.splitext(mel_save_path)
        new_filename = basename + "_noise" + ext 
        
        signal_power = waveform.pow(2).mean()
        snr = 10 ** (db_snr / 10)
        noise_power = signal_power / snr
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
    
        waveform_noisy = waveform + noise  
        
        waveform_np = waveform_noisy.squeeze().cpu().numpy()
        mel_spec = librosa.feature.melspectrogram(y=waveform_np, sr=sr, n_fft=2048, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=(10,4))
        librosa.display.specshow(mel_db)
        plt.savefig(new_filename, format='png')
        plt.close('all')



    if mode == 3:
        basename, ext = os.path.splitext(mel_save_path)
        new_filename = basename + "_pitch" + ext  # poprawne rozszerzenie dla obrazu

        waveform_np = waveform.squeeze().cpu().numpy()  # najpierw konwersja do numpy
    
        y_shifted = librosa.effects.pitch_shift(waveform_np,sr=sr, n_steps=random.uniform(-2, 2))

        mel_spec = librosa.feature.melspectrogram(y=y_shifted, sr=sr, n_fft=2048, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)



        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
        plt.savefig(new_filename, format='png')
        plt.close('all')

    if mode == 4:
        basename, ext = os.path.splitext(mel_save_path)
        new_filename = basename + "_TS" + ext

        waveform_np = waveform.squeeze().cpu().numpy()

        y_stretch = librosa.effects.time_stretch(y = waveform_np, rate = random.uniform(0.3, 1.99))
        mel_spec = librosa.feature.melspectrogram(y=y_stretch, sr=sr, n_fft=2048, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
        plt.savefig(new_filename, format='png')
        plt.close('all')


        




#func for GTZAN to transform raw wav data to melspecs and split them to labeld directories
def create_save_mels_in_directories(input_dir_path, sr, duration, save_dir):

    
    if not os.path.exists(input_dir_path):
        print(f'The directory {input_dir_path} does not exist.')
        return

    for root, dirs, files in os.walk(input_dir_path):
        for name in files:
               
                if name.endswith(".wav"):
                    filename = f"{root}/{name}"
                    mel_name = name.replace(".wav", ".png")
                    genre = root.split(os.sep)[-1]
                    genre_dir_path = f"{input_dir_path}{genre}"
                    mel_save_path = f"{save_dir}/{genre}/{mel_name}"
                
                    print(mel_save_path)
                
                    if not os.path.exists(f"{save_dir}/{genre}"):
                        os.mkdir(f"{save_dir}/{genre}")
                    print(f"Directory created: {f'{save_dir}/{genre}'}")
                   
                if genre == 'pop' or genre == 'blues': 
                        wav_to_mel_with_augmentation(filename, sr, duration, mel_save_path)
                        print(f"File saved in: {genre_dir_path}")
                        continue
                    
# %%
path = '/home/ciona/projects/RCOLM/data/raw_data/GTZAN/genres_original'
save_mel_dir = '/home/ciona/projects/RCOLM/data/converted_data/GTZAN'
sr = 22050
duration = 30
create_save_mels_in_directories(path, sr, duration, save_mel_dir)
                
            


 
