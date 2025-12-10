import numpy as np 


def signal_to_list(signal, sr, num_points=800):
    """Transforms audio signal and returns:
    time - list of time points in seconds
    signal - frequency (0 ... max)
    
    num_points : how long list will be
    """
    sig = np.asarray(signal, dtype=np.float32)
    n = len(sig)
    if n == 0:
        return [], []

    if n <= num_points:
        times = np.linspace(0, n / sr, num=n)
        times = np.round(times, 2).tolist()
        values = np.abs(sig).tolist()
        return times, values

    window = n // num_points
    sig = sig[: window * num_points].reshape(num_points, window)
    env = np.mean(np.abs(sig), axis=1)
    times = np.linspace(0, n / sr, num=num_points)
    times = np.round(times, 2).tolist()
    values = env.tolist()
