import numpy as np
import tempfile
import os
from scipy.io import wavfile
from ligotools.utils import whiten, write_wavfile, reqshift

def test_whiten_and_reqshift():
    fs = 1024
    dt = 1/fs
    t = np.arange(0, 1, dt)
    signal = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)
    psd = lambda f: np.ones_like(f)

    white_signal = whiten(signal, psd, dt)
    assert white_signal.shape == signal.shape
    assert np.isrealobj(white_signal)

    shifted_signal = reqshift(signal, fshift=50, sample_rate=fs)
    assert shifted_signal.shape == signal.shape
    assert np.isrealobj(shifted_signal)


def test_write_wavfile():
    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    filename = tmpfile.name
    tmpfile.close()

    fs = 1024
    t = np.arange(0, 1, 1/fs)
    data = np.sin(2*np.pi*100*t)

    write_wavfile(filename, fs, data)

    fs_read, data_read = wavfile.read(filename)
    assert fs_read == fs
    assert data_read.shape[0] == data.shape[0]

    os.remove(filename)

