# %%
import musdb
import os
from musdb.audio_classes import MultiTrack
from lib.file_save import  Gtzan_db
import soundfile as sf






# %%





# %% [md]
##  Main funkction

# %%   

#variables
musdb_path = "/home/ciona/projects/RCOLM/data/raw_data/musdb18/"


mus = musdb.DB(
    root= musdb_path, 
    subsets=["test"],
    is_wav=False
)
track = mus.tracks[0]  # lub next(t for t in mus if t.name == "Al James - Schoolboy Facination")

print(track.targets.keys())

vocal_audio = track.sources['vocals'].audio
mixture_audio = track.sources['mixture'].audio
sr = track.rate
 
