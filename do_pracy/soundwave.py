import matplotlib.pyplot as plt 
import numpy as np
import wave, sys, os

def visualize(path: str):
    raw = wave.open(path)

    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype="int16")

    f_rate = raw.getframerate()

    time = np.linspace(0, len(signal)/f_rate, num = len(signal))

    plt.figure(1)



    plt.plot(time, signal)
    plt.savefig("waveform.png")
    print("Zapisano wykres do waveform.png")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "music.wav")

    visualize(path)
