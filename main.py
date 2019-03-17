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

if args.inp:

  # fitting model
  svc = fit_SVC(x_train, y_train, _gamma="scale")
  lr = fit_LR(x_train, y_train)
  nb = fit_Bernoulli_NB(x_train, y_train)
  _2nn = fit_2NN(x_train, y_train, _algorithm="ball_tree", _weights="distance")

  file_path = args.inp

  file_lines = open(file_path, "r").read().split("\n")
  del file_lines[0] # remove first lines
  file_lines.remove('') # remove empty lines

  sample_csv = []
  for f in file_lines:
    f = f.replace("female", '1')
    f = f.replace("male", '0')
    f = f.replace('"', '')
    sample_csv.append(f.split(","))

  models = ["SVC", "LR", "NB", "2NN"]

  x_samples = []
  y_samples = []
  for i in range(len(sample_csv)):
    x_samples.append(sample_csv[i][0:-1])
    y_samples.append(sample_csv[i][-1])

  svc_res   = svc.predict(x_samples),
  lr_res    = lr.predict(np.float64(x_samples)),
  nb_res    = nb.predict(np.float64(x_samples)),
  _2nn_res  =_2nn.predict(np.float64(x_samples)),

  # print(svc_res[0])

  success = [0, 0, 0, 0]
  tot = len(svc_res[0])

  print("SVC \t LR \t NB \t 2NN \t label")
  for i in range(tot):
    print(str(svc_res[0][i])+" \t "+str(lr_res[0][i])+" \t "+str(nb_res[0][i])+" \t "+str(_2nn_res[0][i])+" \t "+y_samples[i])

    success[0] += 1 if int(svc_res[0][i])  == int(y_samples[i]) else 0
    success[1] += 1 if int(lr_res[0][i])   == int(y_samples[i]) else 0
    success[2] += 1 if int(nb_res[0][i])   == int(y_samples[i]) else 0
    success[3] += 1 if int(_2nn_res[0][i]) == int(y_samples[i]) else 0

  print(str(success[0])+"/"+str(tot)+" \t "+
        str(success[1])+"/"+str(tot)+" \t "+
        str(success[2])+"/"+str(tot)+" \t "+
        str(success[3])+"/"+str(tot))


