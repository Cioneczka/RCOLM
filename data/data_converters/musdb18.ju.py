# %%
import musdb
import random


def my_function(track):
    '''My fancy BSS algorithm'''

    # get the audio mixture as numpy array shape=(num_sampl, 2)
    track.audio

    # get the mixture path for external processing
    track.path

    # get the sample rate
    track.rate

    # return any number of targets
    estimates = {
        'vocals': vocals_array,
        'accompaniment': acc_array,
    }
    return estimates


# initiate musdb
mus = musdb.DB(root="/home/ciona/projects/RCOLM/data/raw_data/musdb18/"
               )

track = random.choice(mus.tracks)
track.chunk_duration = 5.0
track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
x = track.audio.T           #Transpozycja macierzy 
y = track.targets['vocals'].audio.T     # -""-
print(x)
