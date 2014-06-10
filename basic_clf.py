"""Competition script for Wise2014."""
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score, KFold

print("Loading data from svmlight files.")
X_train, y_train = load_svmlight_file(
    "data/wise2014-train.libsvm", dtype=np.float32, multilabel=True)

X_test, y_test = load_svmlight_file(
    "data/wise2014-test.libsvm", dtype=np.float32, multilabel=True)

print("Binarizing.")
lb = MultiLabelBinarizer()
y_train = lb.fit_transform(y_train)
#http://scikit-learn.org/stable/auto_examples/document_classification_20newsgroups.html
clf = OneVsRestClassifier(LinearSVC(loss='l2', penalty='l2', tol=1e-3,
                                    dual=False), n_jobs=2)

print("Performing cross validation.")
cv = KFold(y_train.shape[0], n_folds=3, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_train, y_train, scoring='f1', cv=cv)
print("CV scores.")
print(scores)
print("F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print("Fitting model.")
clf.fit(X_train, y_train)

print("Predict test set.")
pred_y = clf.predict(X_test)

print("Writing predictions.")
out_file = open("submission.csv", "w")
out_file.write("ArticleId,Labels\n")
nid = 64858
for i in range(pred_y.shape[0]):
    label = list(lb.classes_[np.where(pred_y[i, :] == 1)[0]].astype("int"))
    label = " ".join(map(str, label))
    if label == "":  # If the label is empty, populate the most frequent label
        label = "103"
    out_file.write(str(nid + i) + "," + label + "\n")
out_file.close()
