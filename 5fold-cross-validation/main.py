from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.metrics import make_scorer,accuracy_score,precision_score,recall_score,f1_score
from tabulate import tabulate

iris_data = datasets.load_iris()
X = iris_data.data
Y = iris_data.target
model = svm.SVC()
scores = cross_validate(model, X,Y,cv=5,scoring={
    "accuracy":make_scorer(accuracy_score),
    "precision":make_scorer(precision_score,average="macro"),
    "recall":make_scorer(recall_score,average="macro"),
    "f-score":make_scorer(f1_score,average="macro")
})
contents = []
for keys in scores.keys():
    contents.append([])
    contents[-1].append(keys)
    for x in scores[keys]:
        contents[-1].append(x)
print(tabulate(contents,headers=["Meteric","Fold1","Fold2","Fold3","Fold4","Fold5"],tablefmt='orgtbl'))