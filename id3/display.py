import treelib
dtree = treelib.Tree()

def maketree(tree,parent):
    if(tree== None or tree.value == None):
        return
    global dtree
    try:
        dtree.create_node(str(tree.value),str(parent) + str(tree.value),parent=str(parent))
    except:
        print(parent,tree.value)
    for x in range(len(tree.children)):
        dtree.create_node(str(tree.arrows[x]),str(tree.value) + str(tree.arrows[x]),parent=str(parent) + str(tree.value))
        maketree(tree.children[x], str(tree.value) + str(tree.arrows[x]))

def displaytree(tree):
    global dtree
    dtree.create_node(tree.value,tree.value)
    for x in range(len(tree.children)):
        dtree.create_node(str(tree.arrows[x]),str(tree.value) + str(tree.arrows[x]),parent=str(tree.value))
        maketree(tree.children[x],str(tree.value) + str(tree.arrows[x]))
    dtree.show()