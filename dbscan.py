from libfgraphs import *
from OldExtensionalWeaklyConvexHull import dnhfactory

def expandCluster(g, vertex, _nh, C, eps, minpts):
    vertex['cluster'] = C
    nh = _nh
    for v in nh:
        if v['mark'] == 0:
            v['mark'] = 1
            nhv = set(g.vs.select(dnhfactory(g, v, eps)))
            if len(nhv) >= minpts:
                nh = nh.union(nhv)
        if v['cluster'] == -1:
            v['cluster'] = C
            if v['NOISE']:
                v['NOISE'] = False

def DBSCAN(g, eps=0.5, distfn=GLOBAL_DISTANCE_FN, minpts=5):
    g.vs['mark'] = 0
    g.vs['cluster'] = -1
    g.vs['NOISE'] = False
    C = 0
    for v in g.vs:
        if v['mark'] == 0:
            v['mark'] = 1
            nh = set(g.vs.select(dnhfactory(g, v, eps)))
            if len(nh) < minpts:
                v['cluster'] = -1
                v['NOISE'] = True
            else:
                C = C+1
                expandCluster(g, v, nh, C, eps, minpts)

    return C
