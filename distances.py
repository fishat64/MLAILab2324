import numpy as np
import math
from functools import reduce


def d_minkowski(g, v1, v2, ex=2):
    def adddim(r, l):
        return r+abs(l[0] - l[1])**ex
    return (reduce(adddim, list(zip(v1['coordinates'],v2['coordinates'])), 0))**(1.0/ex)

def d_cosineSimilarity(g, v1, v2):
    dot_product = np.dot(v1['coordinates'], v2['coordinates'])
    norm_point1 = np.linalg.norm(v1['coordinates'])
    norm_point2 = np.linalg.norm(v2['coordinates'])
    if norm_point1 == 0 or norm_point2 == 0:
        return 0
    similarity = dot_product / (norm_point1 * norm_point2)
    return similarity

def d_euclidean(g, v1, v2):
    return d_minkowski(g, v1, v2, 2)

def d_mhOneDim(g, v1, v2, dim):
    return abs(v1['coordinates'][dim]-v2['coordinates'][dim])



def d_manhatten(g, v1, v2):
    return d_minkowski(g, v1, v2, 1)

def d_geodesic(g, v1, v2, key='len'):
    shortestPath = v1.get_shortest_paths(v2, output="epath", weights=g.es[key])
    if shortestPath == []:
        return math.inf
    spl = [reduce(lambda a,e: a+g.es[e][key], shortestPath[i], 0) for i in range(len(shortestPath))]
    return min(spl)

def d_step(g, v1, v2):
    return d_geodesic(g, v1, v2, key='count')

def d_commonNeighbors(g, v1, v2, n_hop=1):
    return len(set(g.neighborhood(v1.index, order=n_hop, mindist=1)).intersection(set(g.neighborhood(v2.index, order=n_hop, mindist=1))))

def d_jaccard(g, v1, v2, n_hop=1):
    divisor = len(set(g.neighborhood(v1.index, order=n_hop, mindist=1)).union(set(g.neighborhood(v2.index, order=n_hop, mindist=1))))
    if divisor == 0:
        return math.inf
    return d_commonNeighbors(g, v1, v2, n_hop=n_hop) / divisor

def d_randomWalk(g, v1, v2, walk_length=20, num_walks=15, retminlen=False):
    def rwincWL(walkl):
        successfullwalksv12 = 0
        unsuccessfulwalksv12 = 0
        succwalklistv12 = []
        for i in range(num_walks):
            walk = g.random_walk(start=v1.index, steps=walkl)
            if v2.index in walk:
                successfullwalksv12 += 1
                succwalklistv12.append(walk)
            else:
                unsuccessfulwalksv12 += 1

        successfullwalksv21 = 0
        unsuccessfulwalksv21 = 0
        succwalklistv21 = []
        for i in range(num_walks):
            walk = g.random_walk(start=v2.index, steps=walkl)
            if v1.index in walk:
                successfullwalksv21 += 1
                succwalklistv21.append(walk)
            else:
                unsuccessfulwalksv21 += 1

        if successfullwalksv12 == 0 or successfullwalksv21 == 0:
            return math.inf
        
        return (successfullwalksv12 + successfullwalksv21) / (successfullwalksv12 + successfullwalksv21 + unsuccessfulwalksv12 + unsuccessfulwalksv21)
    
    cwl = 1
    resrwl = rwincWL(cwl)
    while cwl <= walk_length:
        resrwl = rwincWL(cwl)
        if resrwl == math.inf:
            cwl += 1
        else:
            if retminlen:
                return cwl
            cwl += 1
    return resrwl


def distanceFnFactory(distfn, **kwargs):
    def distfnWrapper(g, v1, v2):
        return distfn(g, v1, v2, **kwargs)
    return distfnWrapper