# -*- coding: utf-8 -*-
"""
Created on Wed May 29 20:29:42 2019

@author: user
"""
import sounddevice as sd
import soundfile as sf
import librosa
from madmom.features.beats import BeatTrackingProcessor
from madmom.features.beats import RNNBeatProcessor
import utils
import matplotlib.pyplot as plt
from audiolazy import lazy_synth
import scipy
import numpy as np
import pdb
import heapq
import pyworld as pw
from praatio.pitch_and_intensity import extractPitch, extractIntensity
from scipy.io.wavfile import write
from scipy.stats import pearsonr
import random
import math
from operator import sub

def cal_beat_samples(all_data, fs):
    fps = librosa.samples_to_frames(fs, hop_length=hop_len, n_fft=win_len)
    fps = 100
    proc = BeatTrackingProcessor(look_aside=0.2, fps=fps)
    act = RNNBeatProcessor()(all_data)
    beat_times = proc(act)
    
    song_len = librosa.samples_to_time(data.shape, sr=fs)[0]
    idx = np.where(beat_times <= song_len)[0]
    new_beat_times = np.zeros(idx.shape)
    new_beat_times[idx] = beat_times[idx]
    beat_samples = librosa.time_to_samples(new_beat_times, sr=fs)
    return beat_samples

def norm_01(x):
    return (x-min(x))/(max(x)-min(x))

def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

def find_n_largelen_cand(result, n):
    cand_len = []
    output = []
    for i in result:
        cand_len.append(len(i))
    cand = heapq.nlargest(n, cand_len)
    cand = list(set(cand))
    for i in cand:
        idx = np.where(i == np.array(cand_len))[0]
        output.extend(np.array(result)[idx])
    #cand = [np.array(i) for i in cand]
    return output

def find_n_smalllen_cand(result, n):
    cand_len = []
    output = []
    for i in result:
        cand_len.append(len(i))
    cand = heapq.nsmallest(n, cand_len)
    cand = list(set(cand))
    for i in cand:
        idx = np.where(i == np.array(cand_len))[0]
        output.extend(np.array(result)[idx])
    #cand = [np.array(i) for i in cand]
    return output

def gen_hihat(all_data, fs, beat_samples, cand):
    hihat = np.zeros(all_data.shape)
    start = librosa.frames_to_samples(cand[0], hop_len, n_fft=win_len)
    end = librosa.frames_to_samples(cand[-1], hop_len, n_fft=win_len)
    cand_len = end-start
    i = 3
    is_hihat = np.zeros(beat_samples.shape)
    while i < len(beat_samples):
        is_hihat[i] = 1
        i = i + 4
    for i, s in enumerate(beat_samples):
        if is_hihat[i] == 1:
            if s+cand_len > hihat.shape:
                break
            hihat[s:s+cand_len] = data[start:end]
    return hihat

def gen_bass(all_data, fs, beat_samples):
    #cand_len = 120000
    cand_len = int((beat_samples[12] - beat_samples[0])*0.8)
    cand_delay_len = int(cand_len/2)
    beat = np.zeros(all_data.shape)
    is_bass = np.zeros(beat_samples.shape)
    group = np.arange(len(beat_samples)) % 16
    down_idx = np.where((group==0))
    is_bass[down_idx] = 1
    repeat_idx = np.where((group==8))
    is_bass[repeat_idx] = 2
    repeat2_idx = np.where((group==12))
    is_bass[repeat2_idx] = 3
    
    for i, s in enumerate(beat_samples):
        if i+4 < beat_samples.shape[0]:
            s1 = beat_samples[i+4]
            if is_bass[i]==1:
                if s1+cand_len > beat.shape:
                    break
                cand = all_data[s:s+cand_len]
                cand2 = all_data[s:s+cand_delay_len]
                beat[s1:s1+cand_len] = cand*0.6
            '''
            elif is_bass[i]==2:
                if s1+cand_delay_len > beat.shape:
                    break
                beat[s1:s1+cand_delay_len] = cand2*0.5
            elif is_bass[i]==3:
                if s1+cand_delay_len > beat.shape:
                    break
                beat[s1:s1+cand_delay_len] = cand2*0.3
            '''
    return beat

def gen_drum(all_data, fs, beat_samples, cand):
    cand_len = len(cand)
    beat = np.zeros(all_data.shape)
    is_drum = np.zeros(beat_samples.shape)
    group = np.arange(len(beat_samples)) % 8
    idx = np.where((group==1) | (group==6))
    is_drum[idx] = 1
    
    for i, s in enumerate(beat_samples):
        if is_drum[i]==1:
            if s+cand_len > beat.shape:
                break
            beat[s:s+cand_len] = cand
    
    add_idx = np.where((group==5))[0]
    add_list = []
    for i, s in enumerate(beat_samples[add_idx]):
        add_list.append(beat_samples[add_idx[i]] + int((beat_samples[add_idx[i]+1]-beat_samples[add_idx[i]])*0.5) )
        add_list.append(beat_samples[add_idx[i]-1] + int((beat_samples[add_idx[i]]-beat_samples[add_idx[i]-1])*0.5) )
    add_list = np.array(add_list)
    
    semi_drum = cand[:int(len(cand)/2)]
    for i, s in enumerate(add_list):
        if s+int(len(cand)/2) > beat.shape:
            break
        beat[s:s+int(len(cand)/2)] = semi_drum
    return beat

praatEXE = 'D:/Acapella-Creator/Praat.exe'
all_song = 'D:/Acapella-Creator/data/lemon.wav'
file = 'D:/Acapella-Creator/data/lemon.wav'
#all_song = 'D:/Acapella-Creator/data/sudden_miss_you_cut.wav'
#file = 'D:/Acapella-Creator/data/sudden_miss_you_cut.wav'
data, fs = librosa.load(file, sr=None, dtype='double')
all_data, fs = librosa.load(all_song, sr=None, dtype='double')

''' Param setting '''
win_len = 2048 # n of fft
hop_len = 512 # samples

rmse = np.log(librosa.feature.rmse(y=data, frame_length=win_len, hop_length=hop_len))


''' frame step to time step'''
time_step = librosa.frames_to_time(range(rmse.shape[-1]), sr=fs, hop_length=hop_len, n_fft=win_len)


''' ZCR, pitch and energy to find candidates for beat'''
zcr = librosa.feature.zero_crossing_rate(data, frame_length=win_len, hop_length=hop_len)
energy = extractIntensity(file, 'D:/Acapella-Creator/data/result/energy.txt', praatEXE,
                          minPitch=65, sampleStep=librosa.samples_to_time(hop_len, fs), 
                          forceRegenerate=True, undefinedValue=0)
pitch = extractPitch(file, 'D:/Acapella-Creator/data/result/pitch.txt', praatEXE,
             sampleStep=librosa.samples_to_time(hop_len, fs), minPitch=65, maxPitch=1047,
                             silenceThreshold=0.01, forceRegenerate=True,
                             undefinedValue=0, medianFilterWindowSize=0,
                             pitchQuadInterp=None)
pitch = np.array(pitch)[:, -1]
energy = np.array(energy)[:, -1]

nor_pitch = norm_01(pitch)
nor_ener = norm_01(energy)
nor_zcr = norm_01(zcr[0,:])
# %% find hihat cand and gen hihat
''' Find samples where pitch==0 and energy>0 '''
idx = np.where(np.bitwise_and(np.bitwise_and(nor_pitch==0, nor_ener>0.6), nor_zcr[1:]>0.25))[0]
result = group_consecutives(idx) # find consecutive samples as candidates for beats
beat_cand = find_n_largelen_cand(result, 5)

rr = beat_cand[np.random.randint(5)]
start = librosa.frames_to_samples(rr[0], hop_len, n_fft=win_len)
end = librosa.frames_to_samples(rr[-1], hop_len, n_fft=win_len)
tmpp = np.concatenate((data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end]), axis=0)
#sd.play(tmpp*10, fs)

beat_samples = cal_beat_samples(all_data, fs)
hihat = gen_hihat(all_data, fs, beat_samples, rr)
# %% find drum cand and gen drum
spectral_novelty = librosa.onset.onset_strength(data, sr=fs)
frames = np.arange(len(spectral_novelty))
t = librosa.frames_to_time(frames, sr=fs)
<<<<<<< HEAD

idx = np.where(np.bitwise_and(nor_pitch>0.2, nor_ener>0.8))[0]
idx = np.where(spectral_novelty[idx]>0.8)[0]
result = group_consecutives(idx)
drum_cand = find_n_largelen_cand(result, 5)
rr = drum_cand[np.random.randint(5)]
start = librosa.frames_to_samples(rr[0]+11, hop_len, n_fft=win_len)
end = librosa.frames_to_samples(rr[-1]+12, hop_len, n_fft=win_len)
tmpp = np.concatenate((data[start:end],data[start:end],data[start:end],data[start:end]), axis=0)
=======
idx = np.where(np.bitwise_and(np.bitwise_and(nor_pitch>0.2, nor_ener>0.8), nor_zcr[1:]<0.2))[0]
idx = np.where(spectral_novelty[idx]>0.8)[0]
# =============================================================================
# idx = np.where(np.bitwise_and(nor_pitch>0.2, nor_ener>0.8))[0]
# idx = np.where(spectral_novelty[idx]>0.85)[0]
# =============================================================================
result = group_consecutives(idx)
drum_cand = find_n_largelen_cand(result, 5)
ii = np.random.randint(5)
rr = drum_cand[ii]
start = librosa.frames_to_samples(rr[0]-5, hop_len, n_fft=win_len)
end = librosa.frames_to_samples(rr[-1]-5, hop_len, n_fft=win_len)
tmpp = np.concatenate((data[start:end],data[start:end]), axis=0)
print(ii)
#sd.play(tmpp*10, fs)
>>>>>>> e02951b817003a193baddd9b18aa58a8ef643548
drum = data[start:end]

_f0, t = pw.dio(drum, fs)    # raw pitch extractor
f0 = pw.stonemask(drum, _f0, t, fs)  # pitch refinement
#f0 = f0[np.where(f0>0)]
sp = pw.cheaptrick(drum, f0, t, fs)  # extract smoothed spectrogram
ap = pw.d4c(drum, f0, t, fs)         # extract aperiodicity
<<<<<<< HEAD
diff = np.mean(f0) - 95
=======
diff = np.mean(f0) - 80
>>>>>>> e02951b817003a193baddd9b18aa58a8ef643548
y = pw.synthesize(f0-diff, sp, ap, fs)
drum = y
adsr = list(lazy_synth.adsr(len(drum), len(drum)*0.1, len(drum)*0.1, len(drum)*0.1, len(drum)*0.7))

drummm = gen_drum(all_data, fs, beat_samples, drum*librosa.util.normalize(adsr))
<<<<<<< HEAD
bass = gen_bass(all_data, fs, beat_samples)
#chorus, fs = librosa.load('../result/lemon_chorus_mix.wav', sr=None)
#%%
n_fft = 100	# (ms)
hop_length = 25	# (ms)

y, sr = librosa.load('../data/lemon.wav', sr=None)
#y, sr = librosa.load('../data/sudden_miss_you_cut.wav', sr=None)
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
#x, fs = librosa.load('../data/sudden_miss_you_cut.wav', dtype='double', sr=None)

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
random2 = random.randrange(round(len(x)*0.4),len(x)-250000)
mix1x = np.zeros((len(x)))
mix1x[random1:random1+250000] = y_up[random1:random1+250000]
mix2x = np.zeros((len(x)))
mix2x[random2:random2+250000] = y_up[random2:random2+250000]
chorus = x + y_down_octave[0:len(x)-len(y_down_octave)]*0.2 + mix1x*0.5 + mix2x*0.5
#sd.play(drummm, fs)
sd.play(drummm*0.8+hihat*0.3+bass*0.8+all_data+chorus, fs)
=======
#sd.play(drummm+hihat, fs)
sd.play(drummm+hihat*1.2+all_data, fs)
>>>>>>> e02951b817003a193baddd9b18aa58a8ef643548
