from functions.import_dataset import *
from functions.new_sample import *
from functions.LR import *

x_train, x_test = normalize_L2(x_train, x_test)
x_train, x_test = PCA_decomposition(x_train, x_test)

# lr = fit_LR(x_train, y_train)
# print("LR")
# predict_and_score(lr, x_test, y_test)

# lr = fit_Bernoulli_NB(x_train, y_train)
# print("NB")
# predict_and_score(lr, x_test, y_test)

# lr = fit_SVC(x_train, y_train, _gamma="scale")
# print("SVC")
# predict_and_score(lr, x_test, y_test)

# lr = fit_2NN(x_train, y_train, _algorithm="ball_tree", _weights="distance")
# print("2NN")
# predict_and_score(lr, x_test, y_test)

lr = fit_SVC(x_train, y_train, _gamma="scale")
print("SVC")
predict_and_score(lr, x_test, y_test)

print(len(x_test[0]))

print (lr.predict([x_test[0]]))
print(y_test[0])

new_sample = []
test = spectral_properties("test.wav")
for t in test:
  new_sample.append(test[t])

print(lr.predict([new_sample]))
