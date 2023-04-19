import pandas
import math
from display import displaytree
import sys

data = pandas.read_csv(sys.argv[1])
feature_values = dict( (x,set(data[x])) for x in  data.columns.tolist())
processed_features = []

class Node:
    def __init__(self):
        self.value = None
        self.children = []
        self.arrows = []

def entropy(data,feature,outputvalues):
    e = 0
    outputcolumn = data.columns.tolist()[-1]
    for x in outputvalues:
        if data.shape[0] != 0:
            probability = float(data[data[outputcolumn] == x].shape[0])/data.shape[0]
            if probability !=0:
                e += probability * math.log(probability,2)
    if e!=0:
        return -e
    else:
        return e

def information_gain(data,feature,values,Es,outputvalues):
    ig = Es
    outputcolumn = data.columns.tolist()[-1]
    for x in values:
        if data.shape[0]!=0:
            ig -= entropy(data[data[feature] == x], feature, outputvalues) * (data[data[feature] == x].shape[0]/data.shape[0])
    return ig

def make_node(data,node):
    global processed_features,feauture_values
    features = data.columns.tolist()
    for x in processed_features:
        features.remove(x)
    outputcolumn = data.columns.tolist()[-1]
    Es = entropy(data,outputcolumn , feature_values[outputcolumn])
    info_gains = {}
    if Es !=0 :
        for x in features[:-1]:
            info_gains[x]= information_gain(data,x,feature_values[x],Es,feature_values[outputcolumn])
        if len(info_gains)!=0:
            maximum = max(info_gains,key=info_gains.get)
            node.value = maximum
            processed_features.append(maximum)
            if(len(processed_features) != len(data.columns.tolist())) :
                for x in feature_values[maximum]:
                    #print("appending",maximum,x)
                    node.arrows.append(x)
                    node.children.append(make_node(data[data[maximum] ==x],Node()))
    else:
        if data[outputcolumn].shape[0] !=0:
            node.value =  list(data[outputcolumn])[0]
    return node

def print_tree(tree):
    if(tree.value !=None):
        print(tree.value)
        for x in range(len(tree.children)):
            print(tree.arrows[x])
            print_tree(tree.children[x])
            print("------",tree.value)

tree = make_node(data,Node())
print_tree(tree)
displaytree(tree)