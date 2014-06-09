import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Reading the files..
print("Loading data from svmlight files.")
X_train, y_train = load_svmlight_file(
    "data/wise2014-train.libsvm", dtype=np.float32, multilabel=True)
X_test, y_test = load_svmlight_file(
    "data/wise2014-test.libsvm", dtype=np.float32, multilabel=True)

print("Binarizing.")
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)

print("Fitting model.")
# Fitting the model and predicting..
clf = OneVsRestClassifier(LogisticRegression(), n_jobs=2)
clf.fit(X_train, y_train)

print("Predict test set.")
pred_y = clf.predict(X_test)

print("Writing predictions.")
# Writing the output to a file..
out_file = open("submission.csv", "w")
out_file.write("ArticleId,Labels\n")
id = 64858
for i in range(pred_y.shape[0]):
    label = list(lb.classes_[np.where(pred_y[i, :] == 1)[0]].astype("int"))
    label = " ".join(map(str, label))
    if label == "":  # If the label is empty, populate the most frequent label
        label = "103"
    out_file.write(str(id+i)+","+label+"\n")
out_file.close()
