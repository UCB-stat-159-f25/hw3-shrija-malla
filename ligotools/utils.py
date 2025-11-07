import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, windows
from scipy.io import wavfile
from matplotlib import mlab
from scipy.interpolate import interp1d

def write_wavfile(filename, fs, data):
    d = np.int16(data / np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename, int(fs), d)

def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    norm = 1./np.sqrt(1./(dt*2))
    hf = np.fft.rfft(strain)
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

def reqshift(data, fshift=100, sample_rate=4096):
    x = np.fft.rfft(data)
    T = len(data)/float(sample_rate)
    df = 1.0/T
    nbins = int(fshift/df)
    y = np.roll(x.real, nbins) + 1j*np.roll(x.imag, nbins)
    y[0:nbins] = 0.
    z = np.fft.irfft(y)
    return z

def plot_psd_and_matched_filter(time, strain_H1, strain_L1, template, fs, dt, figures_dir, eventname, plottype, make_plots=True, fband=[35, 350]):
    NFFT = 4*fs
    psd_window = np.blackman(NFFT)
    NOVL = NFFT // 2
    template_fft = np.fft.fft(template * windows.tukey(template.size, alpha=1./8)) / fs
    datafreq = np.fft.fftfreq(template.size) * fs
    df = np.abs(datafreq[1] - datafreq[0])
    bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
    normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
    for det, strain in zip(['H1', 'L1'], [strain_H1, strain_L1]):
        data_psd, freqs = mlab.psd(strain, Fs=fs, NFFT=NFFT, window=psd_window, noverlap=NOVL)
        data_fft = np.fft.fft(strain * windows.tukey(template.size, alpha=1./8)) / fs
        power_vec = np.interp(np.abs(datafreq), freqs, data_psd)
        optimal = data_fft * template_fft.conjugate() / power_vec
        optimal_time = 2*np.fft.ifft(optimal)*fs
        sigmasq = 1*(template_fft*template_fft.conjugate()/power_vec).sum()*df
        sigma = np.sqrt(np.abs(sigmasq))
        SNR_complex = optimal_time/sigma
        peaksample = int(strain.size/2)
        SNR_complex = np.roll(SNR_complex, peaksample)
        SNR = abs(SNR_complex)
        indmax = np.argmax(SNR)
        timemax = time[indmax]
        SNRmax = SNR[indmax]
        d_eff = sigma / SNRmax
        horizon = sigma/8
        phase = np.angle(SNR_complex[indmax])
        offset = (indmax-peaksample)
        template_phaseshifted = np.real(template*np.exp(1j*phase))
        template_rolled = np.roll(template_phaseshifted, offset)/d_eff
        template_whitened = whiten(template_rolled, interp1d(freqs, data_psd), dt)
        template_match = filtfilt(bb, ab, template_whitened)/normalization
        if make_plots:
            pcolor = 'r' if det=='H1' else 'g'
            strain_whitenbp = filtfilt(bb, ab, strain)/normalization
            plt.figure(figsize=(10,8))
            plt.subplot(2,1,1)
            plt.plot(time-timemax, SNR, pcolor, label=det+' SNR(t)')
            plt.grid('on')
            plt.ylabel('SNR')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.legend(loc='upper left')
            plt.title(det+' matched filter SNR around event')
            plt.subplot(2,1,2)
            plt.plot(time-timemax, SNR, pcolor, label=det+' SNR(t)')
            plt.grid('on')
            plt.ylabel('SNR')
            plt.xlim([-0.15,0.05])
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.legend(loc='upper left')
            plt.savefig(str(figures_dir / f"{eventname}_{det}_SNR.{plottype}"))
            plt.figure(figsize=(10,8))
            plt.subplot(2,1,1)
            plt.plot(time, strain_whitenbp, pcolor, label=det+' whitened h(t)')
            plt.plot(time, template_match, 'k', label='Template(t)')
            plt.ylim([-10,10])
            plt.xlim([-0.15,0.05])
            plt.grid('on')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.ylabel('whitened strain (units of noise stdev)')
            plt.legend(loc='upper left')
            plt.title(det+' whitened data around event')
            plt.subplot(2,1,2)
            plt.plot(time, strain_whitenbp-template_match, pcolor, label=det+' resid')
            plt.ylim([-10,10])
            plt.xlim([-0.15,0.05])
            plt.grid('on')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.ylabel('whitened strain (units of noise stdev)')
            plt.legend(loc='upper left')
            plt.title(det+' Residual whitened data after subtracting template around event')
            plt.savefig(str(figures_dir / f"{eventname}_{det}_matchtime.{plottype}"))
            plt.figure(figsize=(10,6))
            template_f = np.absolute(template_fft)*np.sqrt(np.abs(datafreq))/d_eff
            plt.loglog(datafreq, template_f, 'k', label='template(f)*sqrt(f)')
            plt.loglog(freqs, np.sqrt(data_psd), pcolor, label=det+' ASD')
            plt.xlim(20, fs/2)
            plt.ylim(1e-24,1e-20)
            plt.grid()
            plt.xlabel('frequency (Hz)')
            plt.ylabel('strain noise ASD (strain/rtHz), template h(f)*rt(f)')
            plt.legend(loc='upper left')
            plt.title(det+' ASD and template around event')
            plt.savefig(str(figures_dir / f"{eventname}_{det}_matchfreq.{plottype}"))
