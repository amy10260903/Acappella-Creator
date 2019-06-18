import audioop
import librosa
import librosa.display
import pyworld as pw
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write
from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

file = '../data/lemon.wav'
data, fs = librosa.load(file, dtype='double', sr=None)

_f0, t = pw.dio(data, fs)    # raw pitch extractor
f0 = pw.stonemask(data, _f0, t, fs)  # pitch refinement
sp = pw.cheaptrick(data, f0, t, fs)  # extract smoothed spectrogram
ap = pw.d4c(data, f0, t, fs)         # extract aperiodicity

r = len(data)/len(f0)/fs

plt.close()
plt.figure()
plt.plot(r*np.linspace(0,len(f0),len(f0)),f0/f0.max())

# %%
onset_env = librosa.onset.onset_strength(y=data, sr=fs)
norm_onset = onset_env/onset_env.max()
TH = 0.2
onset = np.where(norm_onset > TH, norm_onset, 0)    
r1 = len(data)/len(onset)/fs
plt.plot(r1*np.linspace(0,len(onset),len(onset)), onset)

#%%
proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
act = RNNDownBeatProcessor()(file)
down_beats_times = proc(act)[:,0]
down_beats = down_beats_times*fs

# %%
import matplotlib.pyplot as plt
y, sr = librosa.load(file, sr=None)
onset_env = librosa.onset.onset_strength(y, sr=sr, aggregate=np.median)
tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
hop_length = 512
plt.figure(figsize=(8, 4))
times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
plt.plot(times, librosa.util.normalize(onset_env), label='Onset strength')
plt.vlines(times[beats], 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')
plt.legend(frameon=True, framealpha=0.75)

# Limit the plot to a 15-second window
plt.xlim(15, 30)
plt.gca().xaxis.set_major_formatter(librosa.display.TimeFormatter())
plt.tight_layout()
# %%
y, sr = librosa.load(file, sr=None)
D = np.abs(librosa.stft(y))
times = librosa.frames_to_time(np.arange(D.shape[1]))
plt.figure()
ax1 = plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
plt.title('Power spectrogram')

onset_env = librosa.onset.onset_strength(y=y, sr=sr)
plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(times, 2 + onset_env / onset_env.max(), alpha=0.8, label='Mean (mel)')

#%%
def time_stretch(signal,alpha):
    '''
    Time stretch signal by a factor alpha. 
    Output will have length alpha times length of the signal.
    '''
    N = int(2**7) 									#frame size
    Hs= int(2**7/8) 								#Synthesis hopsize
    Ha = int(Hs / float(alpha)) 					#Analysis hopsize
    w = np.hanning(2**7) 							#Hanning window
    wscale = sum([i**2 for i in w]) / float(Hs)		#Scaling factor
    L0 = signal.size								#original signal size
    L1 = int(signal.size*alpha)						#output signal size
    
    # Initialization
    signal = np.append(np.zeros(int(len(w))), signal)
    
    time_stretch_signal = np.zeros(int((signal.size)*alpha + signal.size/Ha * alpha))
    idx_synthesis = 0
    idx_analysis = 0
    i = 0
    
    # Loop over frames
    while idx_synthesis <= signal.size - (Hs+N):
        i += 1
        # Get Fourier transform of two consecutive windowed frames    
        X = np.fft.fft(w*signal[idx_synthesis:idx_synthesis+N])
        Xnew = np.fft.fft(w*signal[idx_synthesis+Hs:idx_synthesis+Hs+N])

        if idx_synthesis == 0:
            # First frame is not modified
            Xmod_new = Xnew 

        else:
            # Adapt phase to avoid phase jumps. Xmod contains the modified spectrum of previous frame and 
            # Xmod_new contains the spectrum of the frame we are modifying.
            phi =  np.angle(Xmod) + np.angle(Xnew) - np.angle(X)
            Xmod_new = abs(Xnew) * np.exp(1j*phi)

        #Update parameters
        Xmod = Xmod_new
        time_stretch_signal[idx_analysis:idx_analysis+N] = np.add(time_stretch_signal[idx_analysis:idx_analysis+N], np.real(w*np.fft.ifft(Xmod_new)), 
                                                                  out=time_stretch_signal[idx_analysis:idx_analysis+N], casting="unsafe")
        idx_synthesis = int(idx_synthesis+Ha)		# analysis hop
        idx_analysis += Hs							# synthesis hop

    #Scale and remove tails
    time_stretch_signal = time_stretch_signal / wscale 
    time_stretch_signal = np.delete(time_stretch_signal, range(Hs))
    time_stretch_signal = np.delete(time_stretch_signal, range(L1, time_stretch_signal.size))

    return time_stretch_signal

alpha = 2

pitched_signal = time_stretch(data,alpha)
sd.play(pitched_signal, fs)
write('../result/lemon_stretch.wav', fs, pitched_signal)