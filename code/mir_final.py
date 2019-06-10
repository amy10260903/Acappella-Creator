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
import scipy
import numpy as np
import heapq
import pyworld as pw
from praatio.pitch_and_intensity import extractPitch, extractIntensity

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

def find_cand(result, n):
    cand_len = []
    beat_cand = []
    for i in result:
        cand_len.append(len(i))
    cand = heapq.nlargest(n, cand_len)
    cand = list(set(cand))
    for i in cand:
        idx = np.where(i == np.array(cand_len))[0]
        beat_cand.extend(np.array(result)[idx])
    #beat_cand = [np.array(i) for i in beat_cand]
    return beat_cand

def gen_hihat(all_data, fs, fps, cand):
    fps = librosa.samples_to_frames(fs, hop_length=hop_len, n_fft=win_len)
    fps = 100
    print(cand)
    proc = BeatTrackingProcessor(look_aside=0.2, fps=fps)
    act = RNNBeatProcessor()(all_data)
    beat_times = proc(act)
    
    song_len = librosa.samples_to_time(data.shape, sr=fs)[0]
    hihat = np.zeros(all_data.shape)
    idx = np.where(beat_times <= song_len)[0]
    new_beat_times = np.zeros(idx.shape)
    new_beat_times[idx] = beat_times[idx]
    beat_samples = librosa.time_to_samples(new_beat_times, sr=fs)
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
    
    return hihat, new_beat_times, beat_samples

praatEXE = 'C:/Users/user/Desktop/Praat.exe'
all_song = 'C:/Users/user/Desktop/mir_final/lemon.wav'
file = 'C:/Users/user/Desktop/mir_final/lemon.wav'
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
energy = extractIntensity(file, 'C:/Users/user/Desktop/mir_final/energy.txt', praatEXE,
                          minPitch=65, sampleStep=librosa.samples_to_time(hop_len, fs), 
                          forceRegenerate=True, undefinedValue=0)
pitch = extractPitch(file, 'C:/Users/user/Desktop/mir_final/pitch.txt', praatEXE,
             sampleStep=librosa.samples_to_time(hop_len, fs), minPitch=65, maxPitch=1047,
                             silenceThreshold=0.01, forceRegenerate=True,
                             undefinedValue=0, medianFilterWindowSize=0,
                             pitchQuadInterp=None)
pitch = np.array(pitch)[:, -1]
energy = np.array(energy)[:, -1]


''' Just plot '''
nor_pitch = norm_01(pitch)
nor_ener = norm_01(energy)
nor_zcr = norm_01(zcr[0,:])

tmp_zcr = np.zeros(nor_zcr.shape)
idx = np.where(np.bitwise_and(nor_zcr[1:]>=0.25, nor_ener>0.6))[0]
tmp_zcr[idx] = nor_zcr[idx]

# =============================================================================
# plt.close()
# plt.figure()
# plt.subplot(4,1,1)
# plt.plot(time_step[1:], nor_ener)
# plt.subplot(4,1,2)
# plt.plot(time_step[1:], nor_pitch)
# plt.subplot(4,1,3)
# plt.plot(time_step, nor_zcr.T)
# plt.subplot(4,1,4)
# plt.plot(time_step, tmp_zcr.T)
# =============================================================================
''' Find samples where pitch==0 and energy>0 '''
idx = np.where(np.bitwise_and(np.bitwise_and(nor_pitch==0, nor_ener>0.6), nor_zcr[1:]>0.25))[0]
result = group_consecutives(idx) # find consecutive samples as candidates for beats
''' You can choose some examples in result to hear '''
# =============================================================================
# rr = result[27]
# start = librosa.frames_to_samples(rr[0], hop_len, n_fft=win_len)
# end = librosa.frames_to_samples(rr[-1], hop_len, n_fft=win_len)
# tmpp = np.concatenate((data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end]), axis=0)
# sd.play(tmpp*10, fs)
# =============================================================================

beat_cand = find_cand(result, 5)

rr = beat_cand[2]
start = librosa.frames_to_samples(rr[0], hop_len, n_fft=win_len)
end = librosa.frames_to_samples(rr[-1], hop_len, n_fft=win_len)
tmpp = np.concatenate((data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end],data[start:end]), axis=0)
#sd.play(tmpp*10, fs)
beattt = data[start:end] 

hihat, beat_times, beat_samples = gen_hihat(all_data, fs, 100, beat_cand[2])

# %%
spectral_novelty = librosa.onset.onset_strength(data, sr=fs)
frames = np.arange(len(spectral_novelty))
t = librosa.frames_to_time(frames, sr=fs)
# =============================================================================
# plt.figure(figsize=(15, 4))
# plt.plot(t, spectral_novelty, 'r-')
# plt.xlim(0, t.max())
# plt.xlabel('Time (sec)')
# plt.legend(('Spectral Novelty',))
# =============================================================================

nor_pitch = norm_01(pitch)
nor_ener = norm_01(energy)
nor_zcr = norm_01(zcr[0,:])

idx = np.where(nor_ener>0.8)[0]
idx = np.where(spectral_novelty[idx]>0.8)[0]
result = group_consecutives(idx)
drum_cand = find_cand(result, 8)
rr = drum_cand[2]
start = librosa.frames_to_samples(rr[0]+10, hop_len, n_fft=win_len)
end = librosa.frames_to_samples(rr[-1]+11, hop_len, n_fft=win_len)
tmpp = np.concatenate((data[start:end],data[start:end],data[start:end],data[start:end]), axis=0)
drum = data[start:end]
#sd.play(tmpp, fs)
sd.play(data[start:end], fs)

_f0, t = pw.dio(drum, fs)    # raw pitch extractor
f0 = pw.stonemask(drum, _f0, t, fs)  # pitch refinement
sp = pw.cheaptrick(drum, f0, t, fs)  # extract smoothed spectrogram
ap = pw.d4c(drum, f0, t, fs)         # extract aperiodicity
y = pw.synthesize(f0-120, sp, ap, fs)
drum = y
test = []
sapce = np.zeros(drum.shape)
test.extend(drum)
test.extend(sapce)
test.extend(beattt)
test.extend(sapce)
test.extend(drum)
test.extend(drum[int(0.5*drum.shape[0]):])
test.extend(drum)
test.extend(beattt)
test.extend(sapce)
test.extend(sapce)
test.extend(drum)
test.extend(sapce)
test.extend(beattt)
test.extend(sapce)
test.extend(drum)
test.extend(drum[int(0.5*drum.shape[0]):])
sss = np.zeros(data.shape)
sss[:np.array(test).shape[0]] = np.array(test)
#sd.play(sss*5+data*5, fs)

'''gen drum'''
fps = librosa.samples_to_frames(fs, hop_length=hop_len, n_fft=win_len)
fps = 100
print(fps)
proc = BeatTrackingProcessor(look_aside=0.2, fps=fps)
act = RNNBeatProcessor()(all_data)
beat_times = proc(act)

song_len = librosa.samples_to_time(data.shape, sr=fs)[0]
beat = np.zeros(all_data.shape)
idx = np.where(beat_times <= song_len)[0]
new_beat_times = np.zeros(idx.shape)
new_beat_times[idx] = beat_times[idx]

beat_samples = librosa.time_to_samples(new_beat_times, sr=fs)
cand_len = len(drum)

end = len(beat_samples)
is_drum = np.zeros(beat_samples.shape)
group = np.arange(len(beat_samples)) % 8
idx = np.where((group==1) | (group==6))
is_drum[idx] = 1

for i, s in enumerate(beat_samples):
    if is_drum[i]==1:
        print(s)
        if s+cand_len > beat.shape:
            break
        beat[s:s+cand_len] = drum


add_idx = np.where((group==5))[0]
add_list = []
for i, s in enumerate(beat_samples[add_idx]):
    print(i,s, add_idx[i])
    add_list.append(beat_samples[add_idx[i]] + int((beat_samples[add_idx[i]+1]-beat_samples[add_idx[i]])*0.5) )
    add_list.append(beat_samples[add_idx[i]-1] + int((beat_samples[add_idx[i]]-beat_samples[add_idx[i]-1])*0.5) )
    #add_list.append(int(0.8*(s+beat_samples[add_idx[i]+1])))
add_list = np.array(add_list)

semi_drum = drum[:int(len(drum)/2)]
for i, s in enumerate(add_list):
    print(s)
    if s+int(len(drum)/2) > beat.shape:
        break
    beat[s:s+int(len(drum)/2)] = semi_drum

sd.play(beat*0.5+hihat+data, fs)
rr = drum_cand[3]
start = librosa.frames_to_samples(rr[0]+10, hop_len, n_fft=win_len)
end = librosa.frames_to_samples(rr[-1]+11, hop_len, n_fft=win_len)
tmpp = np.concatenate((data[start:end],data[start:end],data[start:end],data[start:end]), axis=0)
drum = data[start:end]
#sd.play(tmpp, fs)
sd.play(data[start:end], fs)