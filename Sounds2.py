import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pprint


def get_piano_notes():
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
    base_freq = 440  # Frequency of Note A4
    keys = np.array([x + str(y) for y in range(0, 9) for x in octave])
    pprint.pprint(keys)
    # Trim to standard 88 keys
    start = np.where(keys == 'A0')[0][0]
    end = np.where(keys == 'C8')[0][0]
    keys = keys[start:end + 1]

    note_freqs = dict(zip(keys, [2 ** ((n + 1 - 49) / 12) * base_freq for n in range(len(keys))]))
    note_freqs[''] = 0.0  # stop
    return note_freqs


def get_sine_wave(frequency, duration, sample_rate=44100, amplitude=4096):
    t = np.linspace(0, duration, int(sample_rate * duration))  # Time axis
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave


def fast_fourier_transformation(file_wav, plot=False):
    """
    The method fast fourier transformation to extract distinct freq composing
    the provided frequency into its components
    :param file_wav:
    :return:
    """
    sample_rate, middle_c = wavfile.read(file_wav)
    t = np.arange(middle_c.shape[0])
    freq = np.fft.fftfreq(t.shape[-1]) * sample_rate
    sp = np.fft.fft(middle_c)
    if plot:
        # Plot spectrum
        plt.plot(freq, abs(sp.real))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('Spectrum of Middle C Recording on Piano')
        plt.xlim((0, 2000))
        plt.grid()
    return freq, sp


def calculate_ratio_of_magnitude(freq, sp):
    # Get positive frequency
    idx = np.where(freq > 0)[0]
    freq = freq[idx]
    sp = sp[idx]

    # Get dominant frequencies
    sort = np.argsort(-abs(sp.real))[:100]
    dom_freq = freq[sort]

    # Round and calculate amplitude ratio
    freq_ratio = np.round(dom_freq / frequency)
    unique_freq_ratio = np.unique(freq_ratio)
    amp_ratio = abs(sp.real[sort] / np.sum(sp.real[sort]))
    factor = np.zeros((int(unique_freq_ratio[-1]),))
    for i in range(factor.shape[0]):
        idx = np.where(freq_ratio == i + 1)[0]
        factor[i] = np.sum(amp_ratio[idx])
    factor = factor / np.sum(factor)
    return factor


def apply_overtones(frequency, duration, factor, sample_rate=44100, amplitude=4096):

    assert abs(1-sum(factor)) < 1e-8
    frequencies = np.minimum(np.array([frequency*(x+1) for x in range(len(factor))]), sample_rate//2)
    amplitudes = np.array([amplitude*x for x in factor])

    fundamental = get_sine_wave(frequencies[0], duration, sample_rate, amplitudes[0])
    for i in range(1, len(factor)):
        overtone = get_sine_wave(frequencies[i], duration, sample_rate, amplitudes[i])
        fundamental += overtone
    return fundamental


if __name__ == "__main__":
    # Get middle C frequency
    note_freqs = get_piano_notes()
    frequency = note_freqs['C4']

    # Pure sine wave
    sine_wave = get_sine_wave(frequency, duration=2, amplitude=2048)
    wavfile.write('pure_c.wav', rate=44100, data=sine_wave.astype(np.int16))
