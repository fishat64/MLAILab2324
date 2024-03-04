import math
import sys
from distances import d_step, d_geodesic
import numpy as np

import igraph as ig
if "GLOBAL_DISTANCE_FN" not in globals():
    GLOBAL_DISTANCE_FN = d_geodesic

def getMaxDistance(g, distfn=GLOBAL_DISTANCE_FN):
    maxdist = 0
    for v1 in g.vs:
        for v2 in g.vs:
            if v1.index != v2.index:
                dist = distfn(g, v1, v2)
                if dist == math.inf:
                    return sys.maxsize
                if dist > maxdist:
                    maxdist = dist
    return maxdist

def getMinDistance(g, distfn=GLOBAL_DISTANCE_FN):
    mindist = math.inf
    for v1 in g.vs:
        for v2 in g.vs:
            if v1.index != v2.index:
                dist = distfn(g, v1, v2)
                if dist < mindist:
                    mindist = dist
    return mindist

def getAvgDistance(g, distfn=GLOBAL_DISTANCE_FN):
    avgdist = 0
    for v1 in g.vs:
        for v2 in g.vs:
            if v1.index != v2.index:
                dist = distfn(g, v1, v2)
                if dist == math.inf:
                    return sys.maxsize
                avgdist += dist
    return avgdist / (g.vcount()**2 - g.vcount())

def getMaxDistanceToPoint(g, point, distfn=GLOBAL_DISTANCE_FN):
    maxdist = 0
    for v in g.vs:
        dist = distfn(g, v, point)
        if dist == math.inf:
            return sys.maxsize
        if dist > maxdist:
            maxdist = dist
    return maxdist

def getMinDistanceToPoint(g, point, distfn=GLOBAL_DISTANCE_FN):
    mindist = math.inf
    for v in g.vs:
        dist = distfn(g, v, point)
        if dist < mindist:
            mindist = dist
    return mindist

def getAvgDistanceToPoint(g, point, distfn=GLOBAL_DISTANCE_FN):
    avgdist = 0
    for v in g.vs:
        dist = distfn(g, v, point)
        if dist == math.inf:
            return sys.maxsize
        avgdist += dist
    return avgdist / g.vcount()

def getLongestPossiblePath(g):
    return getMaxDistance(g, distfn=d_step)


def indextovertex(g, ind):
    if isinstance(ind, np.number) or np.issubdtype(type(ind), np.number):
        ind = int(ind)
    if isinstance(ind, np.ndarray):
        ind = ind.tolist()
        ind = [int(i) for i in ind]

    if isinstance(ind, ig.VertexSeq):
        return ind
    if isinstance(ind, (list,set)):
        if all([isinstance(i, (int, ig.Vertex)) for i in ind]):
            return [g.vs[i] if isinstance(i, int) else i for i in ind]
        
        raise ValueError("indextovertex: invalid type \n"+str(type(ind))+" "+str(ind))
    if isinstance(ind, ig.Vertex):
        return ind
    if isinstance(ind, int):
        return g.vs[ind]
    print(type(ind), type(g))
    raise ValueError("indextovertex: invalid type \n"+str(type(ind))+" "+str(type(g))+" "+str(ind))

def vertextoindex(v):
    if isinstance(v, np.number) or np.issubdtype(type(v), np.number):
        return int(v)
    if isinstance(v, np.ndarray):
        v = v.tolist()
        v = [int(i) for i in v]

    if isinstance(v, ig.VertexSeq):
        return v.indices
    if isinstance(v, (list,set)):
        if all([isinstance(i, (int, ig.Vertex)) for i in v]):
            return [vi.index if isinstance(vi, ig.Vertex) else vi for vi in v]
        print([type(i) for i in v])
        raise ValueError("vertextoindex: invalid type \n"+str(type(v))+" "+str(v))
    if isinstance(v, ig.Vertex):
        return v.index
    if isinstance(v, int):
        return v
    print(type(v))
    raise ValueError("vertextoindex: invalid type \n"+str(type(v))+" "+str(v))

def save_graph(g, file_name):
    g.write_pickle(file_name)

def load_graph(file_name):
    return ig.Graph().Read_Pickle(file_name)
