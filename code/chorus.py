import pyworld as pw
import sounddevice as sd
import librosa
import numpy as np
import math
from operator import sub
from scipy.io.wavfile import write

x, fs = librosa.load('../data/f1_005.wav', dtype='double', sr=None)


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
    log2temp = [math.log2(i) for i in temp]
    diff = list(map(sub, log2temp, [round(i) for i in log2temp]))
    diff = [abs(i) for i in diff]
    idx = diff.index(min(diff))
    if idx==0 or idx==3 or idx==4:
        chorus[k] = freq_f0*2**(4/12)
    else:
        chorus[k] = freq_f0*2**(3/12)
        
y = pw.synthesize(chorus, sp, ap, fs)
mix = y[0:len(x)-len(y)]*0.6 + x
sd.play(mix, fs)

write('f1_005_chorus_up3.wav', fs, mix)