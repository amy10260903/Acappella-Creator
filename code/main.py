# -*- coding: utf-8 -*-
'''
Created on Sun. Jun 02 2019
@author: Cosine Chen

ASAS Final Project —— Acapella Creator

2019.06.02
Audio Analysis funtions:
1. Pitch Detection
2. Energy Calculation
3. Zero-Crossing Rate
4. Beat Tracking
---
Audio Synthesis functions:
1. Pitch Shifting
2. Time Stretching
3. Reverb/Delay/Echo

[USED PACKAGE]
'''
CHORD = {'C' :[0, 4, 7],  'C#':[1, 5, 8],
		 'D' :[2, 6, 9],  'D#':[3, 7, 10],
		 'E' :[4, 8, 11], 'F' :[5, 9, 0],
		 'F#':[6, 10, 1], 'G' :[7, 11, 2],
		 'G#':[8, 0, 3],  'A' :[9, 1, 4],
		 'A#':[10, 2, 5], 'B' :[11, 3, 6]}
		 
KEY  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

class AudioAnalysis():
	def PitchDetection():
		return ''
	
	def CalEnergy():
		return ''
	
	def ZCR():
		return ''
	
	def TrackBeat():
		return ''
	
	def PitchShift():
		return ''
	
	def TimeStretch():
		return ''
	
	def Delay():
		return ''

	def __init__(self):
		pass

analyzer = AudioAnalysis()
