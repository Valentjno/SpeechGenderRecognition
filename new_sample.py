import sys
import numpy as np

from scipy.stats import gmean
from scipy.io import wavfile

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction

# from aubio import source, pitch

# import librosa
# y, sr = librosa.load("test.wav")
# flatness = librosa.feature.spectral_flatness(y=y)
# print(flatness.mean())

def spectral_properties(file):
  [sample_rate, data] = audioBasicIO.readAudioFile("test.wav");
  F, f_names = audioFeatureExtraction.stFeatureExtraction(data, sample_rate, 0.050*sample_rate, 0.025*sample_rate);

  spec = np.abs(np.fft.rfft(data))
  freq = np.fft.rfftfreq(len(data), d=1 / sample_rate)
  peakf = np.argmax(freq) # unusued
  amp = spec / spec.sum()
  mean = (freq * amp).sum()
  sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
  amp_cumsum = np.cumsum(amp)
  median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
  mode = freq[amp.argmax()]
  Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
  Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
  IQR = Q75 - Q25
  z = amp - amp.mean()
  w = amp.std()
  skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
  kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4
  spec_flatness = gmean(spec**2)/np.mean(spec**2)

  result_d = {
    'meanfreq': mean,
    'sd': sd,
    'median': median,
    'mode': mode,
    'Q25': Q25,
    'Q75': Q75,
    'IQR': IQR,
    'skew': skew,
    'kurt': kurt,
    'centroid': F[3].mean(),
    'sp.ent': F[5].mean(),
    'sfm': spec_flatness,
  }

  return result_d

print(spectral_properties("test.wav"))

