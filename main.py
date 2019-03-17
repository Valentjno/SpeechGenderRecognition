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

if args.inp:

  file_path = args.inp

  file_lines = open(file_path, "r").read().split("\n")
  del file_lines[0] # remove first lines
  file_lines.remove('') # remove empty lines

  sample_csv = []
  for f in file_lines:
    sample_csv.append(f.split(","))

  for i in range(len(sample_csv)):
    new_sample = sample_csv[i][0:-1]

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
        

