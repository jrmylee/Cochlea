import numpy as np

def random_augmentations(x, delay=1000, sr=22050):
    h = np.zeros(delay)
    h[0] = 1
    h[-1] = 1
    x = np.convolve(x, h)
    noise = np.random.normal(0,.005,len(x))
    x = x + noise
    return x.astype('float32')
