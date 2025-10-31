# %%
import musdb
import os
from musdb.audio_classes import MultiTrack
from lib.file_save import  Musdb18_save, Gtzan_db







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

track.targets.keys()

save = Musdb18_save()
for root, dirs, files in os.walk(musdb_path):
    for file_name in files:
        file_path=os.path.join(root, file_name)
        track_id = insert_to_tracks  

        print(f"File {file_path} saved \n")    
    



