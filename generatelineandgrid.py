import igraph as ig
import itertools
from libfgraphs import *


def generateTestLine(numVerticies, numvertcl0=2):
    testgraph = ig.Graph(numVerticies)
    testgraph.vs['coordinates'] = [[i,0] for i in range(numVerticies)]
    testgraph.vs['class'] = [0 for _ in range(numvertcl0)] + [1 for _ in range(numVerticies-(2*numvertcl0))] + [0 for _ in range(numvertcl0)]
    testgraph.vs['mark'] = 0
    testgraph.vs["id"] = generateRandomIds(numVerticies)
    testgraph.vs["color"] = "grey"
    testgraph = classcolor(testgraph, showchanges=False)
    for i in range(numVerticies-1):
        testgraph.add_edge(i,i+1)
    testgraph.es['len'] = 0
    testgraph.es['count'] = 1
    for e in testgraph.es:
        testgraph.es[e.index]['len'] = d_euclidean(testgraph, testgraph.vs[e.source], testgraph.vs[e.target])
    return testgraph

def generateTestGrid(numwidth, numlen, sizecorners):
    testgraph = ig.Graph(numlen*numwidth)

    def calcindex(i,j):
        return i*numwidth+j

    for i,j in itertools.product(range(numlen), range(numwidth)):
        testgraph.vs[calcindex(i,j)]['coordinates'] = [i,j]
        testgraph.vs[calcindex(i,j)]['class'] = 1
        testgraph.vs[calcindex(i,j)]['mark'] = 0

    print(testgraph.vcount())
        
        
    for k,l in itertools.product(range(sizecorners), range(sizecorners)):
        testgraph.vs[calcindex(k,l)]['class'] = 0
        testgraph.vs[calcindex(numlen-1-k,l)]['class'] = 0
        testgraph.vs[calcindex(k,numwidth-1-l)]['class'] = 0
        testgraph.vs[calcindex(numlen-1-k,numwidth-1-l)]['class'] = 0

    testgraph.vs["id"] = generateRandomIds(numlen*numwidth)
    testgraph.vs["color"] = "grey"
    testgraph = classcolor(testgraph, showchanges=False)


    for i,j in itertools.product(range(numlen), range(numwidth)):
        if i-1 >= 0:
            testgraph.add_edge(calcindex(i,j), calcindex(i-1,j))
        if j-1 >= 0:
            testgraph.add_edge(calcindex(i,j), calcindex(i,j-1))
        if i+1 < numlen:
            testgraph.add_edge(calcindex(i,j), calcindex(i+1,j))
        if j+1 < numwidth:
            testgraph.add_edge(calcindex(i,j), calcindex(i,j+1))

    testgraph.simplify()
            
    testgraph.es['len'] = 0
    testgraph.es['count'] = 1
    for e in testgraph.es:
        testgraph.es[e.index]['len'] = d_euclidean(testgraph, testgraph.vs[e.source], testgraph.vs[e.target])
        
    showgraph(testgraph, recursive=False, classmark=0)    
    return testgraph