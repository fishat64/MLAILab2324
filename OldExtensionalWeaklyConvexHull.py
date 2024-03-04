from functools import reduce

import queue
from lib import crdbg
from distances import d_geodesic
from graphfns import *

#gc.set_debug(gc.DEBUG_STATS)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x


RNG = np.random.default_rng(seed=102671797463975) 


GLOBAL_SCALE_FACTOR = 1
GLOBAL_DISTANCE_FN = d_geodesic
def distance_GraphObj(g,v1,v2, distfn=GLOBAL_DISTANCE_FN):
    return distfn(g, v1, v2)

GLOBAL_DIMENSION=2
PRECISION=100

def dnhfactory(g, vertex, delta, distfn=distance_GraphObj):
    def rtfn(v):
        return distfn(g, vertex, v) < delta
    return rtfn

def dnhfactorySE(g, vertex, delta, distfn=distance_GraphObj):
    def rtfn(v):
        return distfn(g, vertex, v) <= delta
    return rtfn
    
def triangleDeltaFactory(g, vx, vy, epsilon=0.0, distfn=distance_GraphObj):
    def rtfn(v):
        if epsilon == 0.0:
            return distfn(g, v, vx) + distfn(g, v, vy) == distfn(g, vx, vy)
        return distfn(g, v, vx) + distfn(g, v, vy) <= distfn(g, vx, vy) + epsilon
    return rtfn

    
def Delta(g,x,y, epsilon=0.1, distfn=distance_GraphObj):
    return set(g.vs.select(triangleDeltaFactory(g, x, y, epsilon=epsilon, distfn=distfn)))

def printset(s):
    idl = []
    for i in s:
        idl.append(id(i))
    print(idl)

def customSetIntersection(A, B):
    return A.intersection(B)

def printvertlist(l, printer=print):
    printer([v.index for v in l])

    
def ExtensionalWeaklyConvexHull(g, vertexset, theta, epsilon=0.0, distfn=distance_GraphObj, debug=False):
    dbgprint = crdbg(_debug=debug)

    #print(dbgprint)

    g.vs['mark'] = 0
    C = set()
    E = set()
    Q = queue.Queue()
    for vertex in vertexset:
        vertex['mark']=1
        Q.put(vertex)
    
    while not Q.empty():
        el = Q.get()
        C.add(el)
        thetanh = set(g.vs.select(dnhfactory(g, el, theta, distfn=distfn)))

        dbgprint("C:")
        printvertlist(C, printer=dbgprint)
        dbgprint("thetanh:")
        printvertlist(thetanh, printer=dbgprint)
        dbgprint("C intersection thetanh:")
        printvertlist(C.intersection(thetanh), printer=dbgprint)


        for nhel in C.intersection(thetanh):
            E.add((el, nhel))
            distElNhel = distfn(g, el, nhel)
            
            nhdistel = set(g.vs.select(dnhfactory(g, el, distElNhel, distfn=distfn)))
            nhdistnhel = set(g.vs.select(dnhfactory(g, nhel, distElNhel, distfn=distfn)))

            dbgprint("distElNhel:")
            dbgprint(distElNhel)
            dbgprint("nhdistel:")
            printvertlist(nhdistel, printer=dbgprint)
            dbgprint("nhdistnhel:")
            printvertlist(nhdistnhel, printer=dbgprint)

            intersect = nhdistel.intersection(nhdistnhel)

            dbgprint("intersect:")
            printvertlist(intersect, printer=dbgprint)

            for z in intersect:
                if not z['mark']==1 and z in Delta(g, el, nhel, epsilon=epsilon, distfn=distfn):
                    z['mark']=1
                    Q.put(z)
    return C,E