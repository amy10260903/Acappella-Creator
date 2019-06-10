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
#import utils
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

def gen_beat(all_data, fs, fps, cand):
    fps = librosa.samples_to_frames(fs, hop_length=hop_len, n_fft=win_len)
    #fps = 100
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
    for s in beat_samples:
        start = librosa.frames_to_samples(beat_cand[2][0], hop_len, n_fft=win_len)
        end = librosa.frames_to_samples(beat_cand[2][-1], hop_len, n_fft=win_len)
        cand_len = end-start
        beat[s:s+cand_len] = data[start:end]
    
    return beat, beat_samples

praatEXE = 'C:/Users/user/Desktop/Praat.exe'
all_song = 'C:/Users/user/Desktop/mir_final/f1_005.wav'
file = 'C:/Users/user/Desktop/mir_final/f1_005.wav'
data, fs = librosa.load(file, sr=None)
all_data, fs = librosa.load(all_song, sr=None)

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

plt.close()
plt.figure()
plt.subplot(4,1,1)
plt.plot(time_step[1:], nor_ener)
plt.subplot(4,1,2)
plt.plot(time_step[1:], nor_pitch)
plt.subplot(4,1,3)
plt.plot(time_step, nor_zcr.T)
plt.subplot(4,1,4)
plt.plot(time_step, tmp_zcr.T)
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
#sd.play(data[start:end], fs)

beat, beat_samples = gen_beat(all_data, fs, 100, beat_cand[2])

sd.play(data*4+beat*2, fs)
