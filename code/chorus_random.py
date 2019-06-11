import pyworld as pw
import sounddevice as sd
import librosa
import numpy as np
import math
from operator import sub
from scipy.io.wavfile import write
import random
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import required_func as rf
import utils # self-defined utils.py file

#x, fs = librosa.load('C:/Users/user/Downloads/Acapella-Creator-master (1)/Acapella-Creator-master/data/lemon.wav', dtype='double', sr=None)

#%%
n_fft = 100	# (ms)
hop_length = 25	# (ms)

sr, y = utils.read_wav('C:/Users/user/Downloads/Acapella-Creator-master (1)/Acapella-Creator-master/data/lemon.wav')
cxx = librosa.feature.chroma_cqt(y=y, sr=sr)

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
x, fs = librosa.load('../data/lemon.wav', dtype='double', sr=None)

_f0, t = pw.dio(x, fs)    # raw pitch extractor
f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
ap = pw.d4c(x, f0, t, fs)         # extract aperiodicity
#y = pw.synthesize(f0*2**(3/12), sp, ap, fs)
#mix = y[0:len(x)-len(y)] + x
#sd.play(mix, fs)

#%% shift to 'C' tonality and generate chorus
tune=1  # 調整到對的大調
f0_C = f0*2**(-(idx_ton-tune)/12)


chorus_up = np.zeros(f0.size)
chorus_down = np.zeros(f0.size)
phonetic = [16.352, 18.354, 20.602, 21.827, 24.5, 27.5, 30.868]  # basic frequencies of phonetic
for k, freq_f0 in enumerate(f0_C):
    if freq_f0==0:
        continue
    temp = freq_f0/phonetic
    log2temp = [math.log2(i) for i in temp]
    diff = list(map(sub, log2temp, [round(i) for i in log2temp]))
    diff = [abs(i) for i in diff]
    idx = diff.index(min(diff))
    if idx==0 or idx==3 or idx==4:
        chorus_up[k] = freq_f0*2**(4/12)
    else:
        chorus_up[k] = freq_f0*2**(3/12)  #升三度
# =============================================================================
    if idx==2 or idx==5 or idx==6:
        chorus_down[k] = freq_f0*2**(-4/12)
    else:
        chorus_down[k] = freq_f0*2**(-3/12)   #降三度
# =============================================================================

chorus_up = chorus_up*2**((idx_ton-tune)/12)
chorus_down = chorus_down*2**(idx_ton/12)
chorus_down_octave = f0/2

y_up = pw.synthesize(chorus_up, sp, ap, fs)
y_down = pw.synthesize(chorus_down, sp, ap, fs)
y_down_octave = pw.synthesize(chorus_down_octave, sp, ap, fs)
#mix = x + y_down_octave[0:len(x)-len(y_down_octave)]*0.2 + y_up[0:len(x)-len(y_up)]*0.5 #+ y_down[0:len(x)-len(y_down)]*0.5
#sd.play(mix, fs)

#################random####################
random1 = random.randrange(0,round(len(x)*0.4))
random2 = random.randrange(round(len(x)*0.4),len(x))
mix1x = np.zeros((len(x)))
mix1x[random1:random1+250000] = y_up[random1:random1+250000]
mix2x = np.zeros((len(x)))
mix2x[random2:random2+250000] = y_up[random2:random2+250000]
mix = x + y_down_octave[0:len(x)-len(y_down_octave)]*0.2 + mix1x*0.5 + mix2x*0.5
print('random1=',random1)

print('random2=',random2)
write('f1_005_chorus_mix5.wav', fs, mix)

