import pandas
from pgmpy.models import BayesianNetwork

data = pandas.read_csv('medical.csv').replace('?', 0).astype(float)
model = BayesianNetwork([
    ('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'),
    ('exang', 'trestbps'), ('trestbps', 'heartdisease'),
    ('fbs', 'heartdisease'), ('heartdisease', 'restecg'),
    ('heartdisease', 'thalach'), ('heartdisease', 'chol')
])
model.fit(data)

cols = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']
inputs = []
for x in cols:
    inputs.append(int(input("Enter your " + x + " :")))
frame = pandas.DataFrame([inputs], columns=cols)

print(model.predict(frame))
