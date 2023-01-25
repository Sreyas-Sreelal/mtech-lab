from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris_data = datasets.load_iris()
train_X,test_X, train_Y,test_Y = train_test_split(iris_data.data,iris_data.target,train_size=0.5)

print("Total train data",len(train_X),"Total test data",len(test_X))
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_X, train_Y)

predicted = knn.predict(test_X)

count_correct = 0
count_wrong = 0
for prediction,expected in zip(predicted,test_Y):
    if prediction == expected:
        print("Correct prediction")
        count_correct += 1
    else:
        print("Wrong prediction")
        count_wrong += 1

print("Wrong Predictions",count_wrong,"Correct Predictions",count_correct)
print("Accuracy(%)",count_correct/len(test_X) * 100)