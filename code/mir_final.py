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
from audiolazy import lazy_synth
import scipy
import numpy as np
import pdb
import heapq
import pyworld as pw
from praatio.pitch_and_intensity import extractPitch, extractIntensity

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
drum = data[start:end]

_f0, t = pw.dio(drum, fs)    # raw pitch extractor
f0 = pw.stonemask(drum, _f0, t, fs)  # pitch refinement
#f0 = f0[np.where(f0>0)]
sp = pw.cheaptrick(drum, f0, t, fs)  # extract smoothed spectrogram
ap = pw.d4c(drum, f0, t, fs)         # extract aperiodicity
diff = np.mean(f0) - 80
y = pw.synthesize(f0-diff, sp, ap, fs)
drum = y
adsr = list(lazy_synth.adsr(len(drum), len(drum)*0.1, len(drum)*0.1, len(drum)*0.1, len(drum)*0.7))

drummm = gen_drum(all_data, fs, beat_samples, drum*librosa.util.normalize(adsr))
#sd.play(drummm+hihat, fs)
sd.play(drummm+hihat*1.2+all_data, fs)