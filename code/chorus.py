import pyworld as pw
import sounddevice as sd
import librosa
import numpy as np

x, fs = librosa.load('C:/Users/AHG/Desktop/ASAS/Final_project/data/lemon.wav', dtype='double', sr=None)


_f0, t = pw.dio(x, fs)    # raw pitch extractor
f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
ap = pw.d4c(x, f0, t, fs)         # extract aperiodicity
#y = pw.synthesize(f0*2**(3/12), sp, ap, fs)
#mix = y[0:len(x)-len(y)] + x
#sd.play(mix, fs)


chorus = np.zeros(f0.size)
phonetic = [16.352, 18.354, 20.602, 21.827, 24.5, 27.5, 30.868]
for k, freq_f0 in enumerate(f0):
    if freq_f0==0:
        continue
    temp = freq_f0/phonetic
    closet_multi = [round(i) for i in temp]
    diff = [abs(j) for j in closet_multi-temp]
    idx = diff.index(min(diff))
    if idx==0 or idx==3 or idx==4:
        chorus[k] = freq_f0*2**(4/12)
    else:
        chorus[k] = freq_f0*2**(3/12)
        
y = pw.synthesize(chorus, sp, ap, fs)
mix = y[0:len(x)-len(y)] + x
sd.play(mix, fs)
