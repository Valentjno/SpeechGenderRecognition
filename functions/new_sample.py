import numpy as np
from scipy.stats import gmean
from scipy.io import wavfile
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction

def spectral_properties(file):
  [sample_rate, data] = audioBasicIO.readAudioFile(file);
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
    'meanfreq': mean/1000,
    'sd': sd/1000,
    'median': median/1000,
    'Q25': Q25/1000,
    'Q75': Q75/1000,
    'IQR': IQR/1000,
    'skew': skew,
    'kurt': kurt,
    'sp.ent': F[5].mean(),
    'sfm': spec_flatness,
    'mode': mode/1000,
    'centroid': F[3].mean()/1000,
  }
  return result_d

def test_new_sample(file):
  new_sample = []
  test = spectral_properties(file)
  new_sample = [test[t] for t in test]

  norm = Normalizer(norm='l2')
  new_sample = norm.transform(np.float64([new_sample]))

  pca=PCA()
  pca.fit(x_train)
  new_sample=pca.transform(new_sample)[0]

  print(svm.predict([new_sample]))
  # print(new_sample)
  return new_sample
