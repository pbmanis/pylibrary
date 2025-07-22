from numpy.random import Generator, PCG64
import pylibrary.tools.digital_filters as DF
import matplotlib.mlab as mlab
import matplotlib.pylab as mpl

def test():
    rng = Generator(PCG64())
    detrend = "linear"
    padding = 16384
    nfft = 2048

    signal = rng.normal(0, 1.0, 100000)
    fs = 100000.0
    LPF = 500.0
    HPF = 100.0
    samplefreq = fs
    fbutter_lpf = DF.SignalFilter_LPFButter(signal, LPF, samplefreq, NPole=8)
    fbessel_lpf = DF.SignalFilter_LPFBessel(signal, LPF, samplefreq, NPole=8)
    fbutter_hpf = DF.SignalFilter_HPFButter(signal, HPF, samplefreq, NPole=8)
    fnotch = DF.NotchFilter(fbutter_lpf, notchf=[250.0], Q=60.0, QScale=False, samplefreq=fs)
    fbp = DF.SignalFilter_Bandpass(signal, HPF, LPF, samplefreq)
    fnotch_zp = DF.NotchFilterZP(fbp, notchf=[250.0], Q=60.0, QScale=False, samplefreq=fs)
    # fbp_zp = DF.SignalFilter_BandpassZP(signal, HPF, LPF, samplefreq)  # not implemented
    downsamp = DF.downsample(signal, 10, axis=0, xvals="subsample")

    display = False
    if display:
        s1 = mlab.psd(
            signal,
            NFFT=nfft,
            Fs=fs,
            detrend=detrend,
            window=mlab.window_hanning,
            noverlap=64,
            pad_to=padding,
        )
        s2 = mlab.psd(
            fbutter_lpf,
            NFFT=nfft,
            Fs=fs,
            detrend=detrend,
            window=mlab.window_hanning,
            noverlap=64,
            pad_to=padding,
        )
        s3 = mlab.psd(
            fbessel_lpf,
            NFFT=nfft,
            Fs=fs,
            detrend=detrend,
            window=mlab.window_hanning,
            noverlap=64,
            pad_to=padding,
        )
        s4 = mlab.psd(
            fbutter_hpf,
            NFFT=nfft,
            Fs=fs,
            detrend=detrend,
            window=mlab.window_hanning,
            noverlap=64,
            pad_to=padding,
        )
        s5 = mlab.psd(
            fnotch,
            NFFT=nfft,
            Fs=fs,
            detrend=detrend,
            window=mlab.window_hanning,
            noverlap=256,
            pad_to=padding,
        )
        f, ax = mpl.subplots(1, 1, figsize=(10, 6))
        ax.plot(s1[1], s1[0], "g-", label="Original Signal")
        ax.plot(s1[1], s2[0], "b-", label="LPF Butterworth")
        ax.plot(s1[1], s3[0], "r-", label="LPF Bessel")
        ax.plot(s1[1], s4[0], "c-", label="HPF Butterworth")
        ax.plot(s1[1], s5[0], "m-", label="Notch Filter")
        ax.legend(fontsize=10)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(0, 2000.)
        mpl.show()
