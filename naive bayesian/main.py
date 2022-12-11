import pandas
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
from collections import defaultdict

encoders = defaultdict(LabelEncoder)
data = pandas.read_csv('bin.csv')
data = data.apply(lambda x:encoders[x.name].fit_transform(x))

output_column = data.columns[-1]
output_values = data[output_column].unique()

def calculateGaussianProbability(x, mean, stdev):
    try:
        expo = math.exp(-(math.pow(float(x) - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * expo
    except:
        return 0

def predict(train,inputdata):
    probabilities = {}
    for x in output_values:
        probabilities[x] = 1
        sample = train[train[output_column] == x]
        means = train[train[output_column] == x].mean()[:-1]
        stds = train[train[output_column] == x].std()[:-1]
        for z in sample.columns[:-1]:
            probabilities[x] *= calculateGaussianProbability(inputdata[z],means[z],stds[z])
    print("Predicting ",encoders[output_column].inverse_transform([max(probabilities,key=probabilities.get)])[0])
        
def calculateAccuracy(train,test):
    correct = 0
    means = {}
    stds = {}
    y_pred,y_test = [],[]

    for x in output_values:
        means[x] = train[train[output_column] == x].mean()[:-1]
        stds[x] = train[train[output_column] == x].std()[:-1]
    for _,row in test.iterrows():
        y_test.append(row[output_column])
        probabilities = {}
        for x in output_values:
            sample = train[train[output_column] == x]
            for z in sample.columns[:-1]:
                if x in probabilities:
                    probabilities[x] *= calculateGaussianProbability(row[z],means[x][z],stds[x][z])
                else:
                    probabilities[x] = calculateGaussianProbability(row[z],means[x][z],stds[x][z]) 
        y_pred.append(max(probabilities,key=probabilities.get))
        if(max(probabilities,key=probabilities.get) == row[output_column]):
            
            correct+=1
    cm = confusion_matrix(y_test, y_pred)
    print()
    print(classification_report(y_test, y_pred))
    print("precision: ",cm[0][0]/(cm[0][0]+cm[1][0]))
    return correct/test.shape[0] * 100.0
data = data.sample(n=1000)
test = data.sample(frac=0.33)
train = data.drop(test.index)
print("testdata : ",test.shape[0],"traindata: ",train.shape[0])
print("Accuracy: ",calculateAccuracy(train, test))
print("Training completed, enter data to predict: ")
"""
inputdata = {'pregnancies': ['0'], 'glucose': ['148'], 'blood pressure': ['72'], 'skin thickness': ['35'], 'insulin': ['0'], 'bmi': ['33.6'], 'diabetes pedigree function': ['0.627'], 'age': ['50']}

for x in train.columns[:-1]:
    inputdata[x] = [input("Enter "+x + ": ")]

print(inputdata)
inputdata = pandas.DataFrame(inputdata).apply(lambda x: encoders[x.name].transform(x))

predict(train, inputdata)

"""