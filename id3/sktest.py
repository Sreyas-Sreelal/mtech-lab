from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import pandas
import graphviz

data = pandas.read_csv('tennis.csv')

label = LabelEncoder()
for x in data[:]:
    print(x)
    data[x] = label.fit_transform(data[x])

classifier = tree.DecisionTreeClassifier(criterion="entropy")
Y = data['answer']
X= data.drop(['answer'],axis=1)
classifier = classifier.fit(X,Y)
#tree.plot_tree(classifier)
graph = graphviz.Source(tree.export_graphviz(classifier,out_file='test'))
print(graph)
