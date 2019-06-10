import pyworld as pw
import sounddevice as sd
import librosa
from librosa.feature import chroma_stft, chroma_cqt, chroma_cens
import numpy as np
import math
from operator import sub
from scipy.io.wavfile import write
from scipy.stats import pearsonr
#import matplotlib.pyplot as plt
import required_func as rf
import utils # self-defined utils.py file

#%%
n_fft = 100	# (ms)
hop_length = 25	# (ms)

sr, y = utils.read_wav('C:/Users/AHG/Desktop/ASAS/Final_project/data/lemon.wav')
cxx = chroma_cqt(y=y, sr=sr)

#%% find tonality
chroma_vector = np.sum(cxx, axis=1)
key_ind = np.array(utils.KEY)[np.argmax(chroma_vector)]

major = utils.rotate(utils.MODE['major'], np.argmax(chroma_vector))
minor = utils.rotate(utils.MODE['minor'], np.argmax(chroma_vector))
major_cor = pearsonr(major, chroma_vector)[0]
minor_cor = pearsonr(minor, chroma_vector)[0]
if major_cor > minor_cor:
    mode = 'major'
else:
    mode = 'minor'
#print(key_ind, mode)
KEY = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
idx_ton = KEY.index(key_ind)  # shift number

#%% vocoder analysis
x, fs = librosa.load('C:/Users/AHG/Desktop/ASAS/Final_project/data/lemon.wav', dtype='double', sr=None)

_f0, t = pw.dio(x, fs)    # raw pitch extractor
f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
ap = pw.d4c(x, f0, t, fs)         # extract aperiodicity
#y = pw.synthesize(f0*2**(3/12), sp, ap, fs)
#mix = y[0:len(x)-len(y)] + x
#sd.play(mix, fs)

#%% shift to 'C' tonality and generate chorus
f0_C = f0*2**(-idx_ton/12)


chorus = np.zeros(f0.size)
phonetic = [16.352, 18.354, 20.602, 21.827, 24.5, 27.5, 30.868]  # basic frequencies of phonetic
for k, freq_f0 in enumerate(f0_C):
    if freq_f0==0:
        continue
    temp = freq_f0/phonetic
    log2temp = [math.log2(i) for i in temp]
    diff = list(map(sub, log2temp, [round(i) for i in log2temp]))
    diff = [abs(i) for i in diff]
    idx = diff.index(min(diff))
#    if idx==0 or idx==3 or idx==4:
#        chorus[k] = freq_f0*2**(4/12)
#    else:
#        chorus[k] = freq_f0*2**(3/12)  #升三度
    if idx==2 or idx==5 or idx==6:
        chorus[k] = freq_f0*2**(-4/12)
    else:
        chorus[k] = freq_f0*2**(-3/12)   #降三度

chorus = chorus*2**(-idx_ton/12)

y = pw.synthesize(chorus, sp, ap, fs)
mix = y[0:len(x)-len(y)]*0.6 + x
#sd.play(mix, fs)

#write('f1_005_chorus_up3.wav', fs, mix)
#plt.figure()
#r = len(x)/len(f0)/fs
#plt.plot(r*np.linspace(0,len(f0),len(f0)),f0)

