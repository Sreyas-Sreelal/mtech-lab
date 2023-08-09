class Node:
    def __init__(self,name):
        self.name = name
        self.links = []
        self.outdegree = 0

    def link(self,node):
        self.links.append(node)
        self.outdegree += 1

def pagerank(nodes,key):
    rank = 0
    default_rank = 1/len(nodes)

    for x in nodes.keys():
        if x == key:
            continue
        if nodes[key] in nodes[x].links:
            rank += default_rank/nodes[x].outdegree
    return rank            

nodes = dict([(chr(x+65),Node(chr(x+65))) for x in range(int(input("Number of nodes: ")))])
print(nodes)
print("Type node linkage x to y in format x y and write done to finish linkage\n")

while True:
    userinput = input("")
    
    if userinput == "done":
        break
    X,Y = userinput.split()
    nodes[X].link(nodes[Y])
while True:
    try:
        query = input("Enter name of the node to calculate it's web page rank (Ctrl+C to quit!): ")
        print("Page rank of",query,pagerank(nodes,query))
    except KeyError:
        print("Node doesn't exists!")
    except KeyboardInterrupt:
        exit(0)





