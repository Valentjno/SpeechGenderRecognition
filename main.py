from functions.import_dataset import *
from functions.new_sample import *
from functions.LR import *

x_train, y_train, x_test, y_test = imp_dataset("dataset/voice.csv")

# normalization & PCA decomposition
x_train, x_test = normalize_L2(x_train, x_test)
x_train, x_test = PCA_decomposition(x_train, x_test)

# run test
# run_classifier(x_train, y_train, x_test, y_test)

svc = fit_SVC(x_train, y_train, _gamma="scale")
acc, conf_matrix = predict_and_score(svc, x_test, y_test)
print("SVC: ", acc)

new_sample = []
test = spectral_properties("valentino.wav")
for t in test:
  new_sample.append(test[t])
print(new_sample)
print(svc.predict([new_sample]))
