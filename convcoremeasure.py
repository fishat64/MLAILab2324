from libfgraphs import d_geodesic, showgraph, ShowNumAttr, d_euclidean, getMaxDistance
from OldExtensionalWeaklyConvexHull import dnhfactory, ExtensionalWeaklyConvexHull
from queue import queue
import itertools
import numpy as np
from Measures import convSanity1, convSanity2sampled

GLOBAL_DISTANCE_FN = d_geodesic
def distance_GraphObj(g,v1,v2, distfn=GLOBAL_DISTANCE_FN):
    return distfn(g, v1, v2)

def NewtriangleDeltaFactory(g, vx, vy, epsilon=0.0, theta=1.0, distfn=distance_GraphObj):
    def rtfn(v):
        if epsilon <0.0 or epsilon > 1.0:
            raise("epsilon must be between 0 and 1")
        return (distfn(g, v, vx) + distfn(g, v, vy) <= distfn(g, vx, vy) + epsilon) and (distfn(g, v, vx) + distfn(g, v, vy) >= distfn(g, vx, vy) - epsilon)
    return rtfn

def NewDelta(g,x,y, epsilon=0.0, theta=1.0, distfn=distance_GraphObj):
    return set(g.vs.select(NewtriangleDeltaFactory(g, x, y, epsilon=epsilon, theta=1.0, distfn=distfn)))

def convcore(g, vertexSet, distfn=distance_GraphObj):
    orgset = set(vertexSet)
    components = []
    def triangleEquality(g, v1, v2, v3):
        v1v2 = distfn(g, v1, v2)
        v2v3 = distfn(g, v2, v3)
        v1v3 = distfn(g, v1, v3)
        if v1v2 + v2v3 == v1v3:
            return True
        if v1v2 + v1v3 == v2v3:
            return True
        if v2v3 + v1v3 == v1v2:
            return True
        return False
    
    while True:
        component = set()
        workqueue = queue.Queue()
        for v1,v2,v3 in itertools.combinations(list(orgset), 3):
            if triangleEquality(g, v1, v2, v3):
                component.add(v1)
                component.add(v2)
                component.add(v3)
                orgset = orgset.difference(component)
                workqueue.put(v1)
                workqueue.put(v2)
                workqueue.put(v3)
                break 

        while not workqueue.empty():
            print(workqueue.qsize())
            v1 = workqueue.get()
            if workqueue.empty():
                break
            v2 = workqueue.get()
            for v3 in orgset:
                if triangleEquality(g, v1, v2, v3):
                    component.add(v3)
                    orgset = orgset.difference(component)
                    workqueue.put(v3)

        components.append(component)
        if not any(triangleEquality(g, v1, v2, v3) for v1,v2,v3 in itertools.combinations(list(orgset), 3)):
            break

    return components

def convcore2(g, vertexSet):
    orgset = set(vertexSet)
    components = []
    while True:
        component = set()
        for v1, v2 in itertools.combinations(list(orgset), 2):
            spl = g.get_shortest_paths(v1.index, to=v2.index, output="vpath")
            for sp in spl:
                if len(sp) == 2 and len(component) == 0:
                    component.add(v1)
                    component.add(v2)
                    orgset = orgset.difference(component)
                    break 
            if len(component) == 2:
                break
            
        for v in orgset:
            verticiesInComponent = queue.Queue()
            for cv in component:
                verticiesInComponent.put(cv)
            
            while not verticiesInComponent.empty():
                cv = verticiesInComponent.get()
                spl = g.get_shortest_paths(v.index, to=cv.index, output="vpath")
                spl = [g.vs[sp] for sp in spl]
                if any([ set(sp[1:]).issubset(component) for sp in spl ]):
                    component.add(v)
                    break

                
        orgset = orgset.difference(component)
        
        components.append(component)
        if not any(len(sp) == 2 for sp in g.get_shortest_paths(v1.index, to=v2.index, output="vpath") for v1,v2 in itertools.combinations(list(orgset), 2)):
            break
    return components
                    
                
        
                    



def convmeasure(g, vertexset, theta=1.0, epsilon=0.0, distfn=distance_GraphObj):
    maxdist = max([distfn(g, v1, v2) for v1,v2 in itertools.combinations(g.vs, 2)])
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
        thetanh = set(g.vs.select(dnhfactory(g, el, theta*maxdist, distfn=distfn)))
        for nhel in C.intersection(thetanh):
            E.add((el, nhel))
            distElNhel = distfn(g, el, nhel)
            
            nhdistel = set(g.vs.select(dnhfactory(g, el, distElNhel, distfn=distfn)))
            nhdistnhel = set(g.vs.select(dnhfactory(g, nhel, distElNhel, distfn=distfn)))

            intersect = nhdistel.intersection(nhdistnhel)
            for z in intersect:
                if not z['mark']==1 and z in NewDelta(g, el, nhel, theta=theta*maxdist, epsilon=epsilon, distfn=distfn):
                    z['mark']=1
                    Q.put(z)

    return C,E

def multipleapplication(g, n=3):
    classes = g.vs['class']
    for i in range(n):
        convmeasure(g, g.vs.select(class_eq=0), theta=1, epsilon=0.001, distfn=d_euclidean)
        showgraph(g, recursive=False)
        g.vs['class'] = [1- mark for mark in g.vs['mark']]
    g.vs['class'] = classes

def convcoremeasure(g):
    ConvCore = convcore(g, g.vs.select(class_eq=0), distfn=d_geodesic)
    ConvCore = convcore2(g, g.vs.select(class_eq=0))
    print([len(core) for core in ConvCore])
    
    def placein(v):
        for i, core in enumerate(ConvCore):
            if v in core:
                return i
        return -1
    
    for v in g.vs:
        v['cluster'] = placein(v)

    ShowNumAttr(g, attr='cluster')

def logarithmically_distributed_list(n, min_value=0.01, max_value=1.0):
    if n < 2:
        raise ValueError("The list length 'n' must be at least 2.")
    log_spaced_values = np.logspace(np.log10(min_value), np.log10(max_value), n)
    log_spaced_values[0] = min_value
    log_spaced_values[-1] = max_value

    return log_spaced_values

def trymeasure(g, A):
    lengthepsilon = 10
    lengththeta = 10
    epsilonvals = np.concatenate([[0],logarithmically_distributed_list(lengthepsilon-1, min_value=0.0001, max_value=1.0)])
    thetavals = np.concatenate([[0],logarithmically_distributed_list(lengththeta-1, min_value=0.0001, max_value=getMaxDistance(g))])
    print(epsilonvals, thetavals)
    res = []
    workingpairs = []
    maxepsilon = 0
    maxtheta = 0
    for epsilon in epsilonvals:
        for theta in thetavals:
            AConvNow = ExtensionalWeaklyConvexHull(g, A, theta, epsilon=epsilon)[0]
            res.append((epsilon, theta, len(A), len(AConvNow), convSanity1(A, AConvNow), convSanity2sampled(g, A, AConvNow, samplepercentage=0.1)))
            if len(A)!=len(AConvNow):
                pass
                #print(maxepsilon, maxtheta)
            else:
                oldmaxepsilon, oldmaxtheta = maxepsilon, maxtheta
                if not any([x[0]==epsilon and x[1]==theta for x in workingpairs]):
                    workingpairs.append((epsilon, theta))

                if epsilon > maxepsilon:
                    maxepsilon = epsilon
                    print(oldmaxepsilon, oldmaxtheta)
                if theta > maxtheta:    
                    maxtheta = theta
                    print(oldmaxepsilon, oldmaxtheta)
    return workingpairs
