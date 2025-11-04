# %%

import crepe
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


data_matrix = [[],[]]

sr, audio = wavfile.read('/home/ciona/projects/RCOLM/data/raw_data/GTZAN/genres_original/classical/classical.00000.wav')
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)

if len(time) == len(frequency):
    for t, f in zip(time, frequency):
            data_matrix[0].append(t)
            data_matrix[1].append(f)

#data_frame = pd.DataFrame(data_matrix[0], data_matrix[1])

fig, ax = plt.subplots()
ax.plot(data_matrix[0],data_matrix[1])
ax.set(xlabel ='time', ylabel = 'frequency',
       title = 'Reprezentacja linni melodycznej stworzonej przez CREPE')
ax.grid()
plt.show()



