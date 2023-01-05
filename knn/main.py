from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris_data = datasets.load_iris()
ratio = int(len(iris_data.data)* 0.90)
train_X = iris_data.data[:ratio]
test_X = iris_data.data[ratio:]
train_Y = iris_data.target[:ratio]
test_Y = iris_data.target[ratio:]
print("Total train data",len(train_X),"Total test data",len(test_X))
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_X, train_Y)
predicted = knn.predict(test_X)

"""input_X = []
for feature in iris_data.feature_names:
    input_X.append(float(input("Enter " + feature+ ": ")))
"""
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