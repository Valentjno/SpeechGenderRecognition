from functions.import_dataset import *
from functions.new_sample import *
from functions.LR import *
import argparse

parser = argparse.ArgumentParser(description='car model recognition')

parser.add_argument("-i", "--input",      action="store",       dest="inp",   help="Take a sample wav file and classify it", type=str)
parser.add_argument("-r", "--run",        action="store_true",                help="Run the classifier and see the accuracy results")
args = parser.parse_args()


x_train, y_train, x_test, y_test = imp_dataset("dataset/voice.csv")

# normalization & PCA decomposition
# x_train, x_test = normalize_L2(x_train, x_test)
# x_train, x_test = PCA_decomposition(x_train, x_test)

# run test
if args.run: # --run or -r
  run_classifier(x_train, y_train, x_test, y_test)

svc = fit_SVC(x_train, y_train, _gamma="scale")
acc, conf_matrix = predict_and_score(svc, x_test, y_test)
# print("SVC: ", acc)

def test_new_sample(file):
  new_sample = []
  test = spectral_properties(file)
  new_sample = [test[t] for t in test]

  from sklearn.preprocessing import Normalizer
  norm = Normalizer(norm='l2')
  new_sample = norm.transform([new_sample])

  from sklearn.decomposition import PCA
  pca=PCA()
  pca.fit(x_train)
  new_sample=pca.transform(new_sample)[0]

  print(svc.predict([new_sample]))
  # print(new_sample)
  return new_sample

example = [
  '0.190680448873961,0.0611377025545973,0.218570254724733,0.125543686661189,0.236132566420159,0.11058887975897,2.53557123149387,9.72125880684744,0.906468587834449,0.42622713613279,0.215349219391947,0.190680448873961,0.11193791079686,0.0157946692991115,0.266666666666667,0.673092532467532,0,5.84375,5.84375,0.119661563255439,"male"'.split(","),
  '0.164631821707416,0.0614942090044271,0.17532281205165,0.106456241032999,0.209239598278336,0.102783357245337,1.63076192172708,6.08142469210075,0.944809322109951,0.512734101889552,0.0919942611190818,0.164631821707416,0.105192515171571,0.0158259149357072,0.275862068965517,0.366060697115385,0,6.2890625,6.2890625,0.0781974901761947,"male"'.split(","),
  '0.127391523853866,0.0815297034253743,0.100200668896321,0.0525266038309517,0.212745515354211,0.160218911523259,1.90057117620629,8.27312820109929,0.959985763504569,0.677452303256416,0.0375433262389784,0.127391523853866,0.127546439842296,0.0160804020100502,0.275862068965517,0.116071428571429,0.0234375,0.5390625,0.515625,0.137037037037037,"female"'.split(","),
  '0.206567693461897,0.0419344301762503,0.207542087542088,0.189360269360269,0.229360269360269,0.04,2.66250705005288,11.472257083534,0.882451589678232,0.306095601737118,0.190572390572391,0.206567693461897,0.175525047525782,0.0172786177105832,0.271186440677966,0.250664893617021,0,0.703125,0.703125,0.173015873015873,"female"'.split(","),
]

for i in range(len(example)):
  new_sample = example[i][0:-1]

  # from sklearn.preprocessing import Normalizer
  # norm = Normalizer(norm='l2')
  # new_sample = norm.transform([new_sample])

  # from sklearn.decomposition import PCA
  # pca=PCA()
  # pca.fit(x_train)
  # new_sample=pca.transform(new_sample)[0]

  #print(new_sample)
  print(svc.predict([new_sample]))

  # for i in range(len(x_train[0])):
  #   print(x_train[0][i], "\t\t", new_sample[i])
      

