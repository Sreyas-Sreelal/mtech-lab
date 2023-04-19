from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn import svm

iris_data = datasets.load_iris()
X = iris_data.data
Y = iris_data.target
model = svm.SVC()

scores = cross_validate(model, X,Y,cv=5,scoring=['precision_macro','accuracy','recall_macro','f1_macro'])
for idx in range(5):
    print("Fold",idx+1)
    print("Precision",scores['test_precision_macro'][idx])
    print("Accuracy",scores['test_accuracy'][idx])
    print("Recall",scores['test_recall_macro'][idx])
    print("F1 Score",scores['test_f1_macro'][idx])
    print()
