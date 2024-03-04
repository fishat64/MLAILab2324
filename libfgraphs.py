import igraph as ig
import numpy as np
from scipy.spatial import Delaunay
import math
import matplotlib.pyplot as plt
import plotly.express as px
from functools import reduce
import sys
import queue
import copy
from lib import SplitList, createTaskQueue, createMPPool, processaftereachother, printdebug, crdbg, UnionFind, AreaCache, TimeMeasure, parallelFun, ifprint, samplepairs
from distances import d_geodesic, d_euclidean, distanceFnFactory, d_mhOneDim
from graphfns import *
import itertools
import uuid
import pandas as pd

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


def generateRandomCoordinates(n, cscale=1):
    global GLOBAL_DIMENSION
    global GLOBAL_SCALE_FACTOR
    global PRECISION
    coordinates = RNG.random(size=(n, GLOBAL_DIMENSION))*GLOBAL_SCALE_FACTOR*cscale
    coordinates = np.around(coordinates, decimals=PRECISION)
    return coordinates.tolist()

def generateRandomCoordinatesShifted(n, cscale=1, shift=-0.5):
    coordinates = np.array(generateRandomCoordinates(n, cscale=cscale), dtype=np.float64)
    coordinates = coordinates + shift
    return coordinates.tolist()


def generateRandomCoordinatesGaussian(n, cscale=1):
    global GLOBAL_DIMENSION
    global GLOBAL_SCALE_FACTOR
    global PRECISION
    coordinates = RNG.normal(size=(n, GLOBAL_DIMENSION))*GLOBAL_SCALE_FACTOR*cscale
    coordinates = np.around(coordinates, decimals=PRECISION)
    return coordinates.tolist()

def generateRandomIds(n):
    ids = []
    for _ in range(n):
        newid = str(uuid.uuid4())
        while newid in ids:
            newid = str(uuid.uuid4())
        ids.append(newid)
    return ids
""" Color the graph according to the class attribute """
def classcolor(g, showchanges=True, classmark=0):
    if showchanges:
        colordict = {0: "red", 1: "blue", 2:"orange", -1:"grey"}
        g.vs['color'] = [colordict[ 0 if vclass==classmark else ( 2 if ( not vclass==classmark ) and vmark==1 else 1 )] for vclass,vmark in zip(g.vs["class"], g.vs["mark"])]
        return g
    g.vs["color"]=  ['red' if vclass==0 else ('blue' if vclass==1 else 'grey') for vclass in g.vs['class']]
    return g

""" create a graph with n vertices """
def generateRandomPointCloud(n, coordinateGenerator=generateRandomCoordinates, cscale=1):
    g = ig.Graph(n=n, edges=[])
    g.vs['coordinates'] = coordinateGenerator(n, cscale=cscale)
    g.vs['class'] = -1
    g.vs['mark'] = 0
    g.vs["color"] = "grey"
    g.vs["id"] = generateRandomIds(n)

    g.es['len'] = 0
    g.es['count'] = 1
    g=classcolor(g, showchanges=False)
    return g

""" create a graph with n vertices and edges according to the delaunay triangulation """
def generateRandomGraph(n, coordinateGenerator=generateRandomCoordinates, deleteEdges=0.05, cscale=1):
    g = ig.Graph(n)
    g.vs['coordinates'] = coordinateGenerator(n, cscale=cscale)
    g.vs['class'] = -1
    g.vs['mark'] = 0
    g.vs["color"] = "grey"
    g.vs["id"] = generateRandomIds(n)

    layout = ig.Layout(coords=g.vs['coordinates'])
    delaunay = Delaunay(layout.coords)
    for tri in delaunay.simplices:
        g.add_edges([
            (tri[0], tri[1]),
            (tri[1], tri[2]),
            (tri[0], tri[2]),
        ])
    g.simplify()
    g.es['len'] = 0
    g.es['count'] = 1
    for e in g.es:
        g.es[e.index]['len'] = d_euclidean(g, g.vs[e.source], g.vs[e.target])
    
    if deleteEdges > 0:
        edgeindicies = g.es.indices
        sortededges = sorted(edgeindicies, key=lambda x: g.es[x]['len'], reverse=True)
        g.delete_edges(sortededges[:math.floor(g.ecount()*deleteEdges)])
    return g


""" plot the graph """
def showgraph(g, recursive=True, classmark=0, changecolor=True, saveAs=None, text="", noshow=False):
    print(text, len(g.vs.select(class_eq=0)), len(g.vs.select(class_eq=1)))
    shape_dict = {1: "circle", 0: "rectangle", 2:"triangle-up"}
    g.vs["shape"] = [shape_dict[vmark] for vmark in g.vs["mark"]]

    oldcolor = g.vs["color"]
    if changecolor:
        g=classcolor(g, showchanges=True, classmark=classmark)
    
    #g.es["label"] = g.es["len"]
    g.vs["label"] = [str(v.index) for v in g.vs]
    fig, ax = plt.subplots()
    ig.plot(
        g,
        layout=ig.Layout(coords=g.vs['coordinates']),
        target=ax,
        vertex_size=7,
        edge_width=0.7,
        bbox=(1000,1000)
    )

    if not noshow:
        plt.show()
        print(g.vs.select(class_eq=classmark, mark_eq=1).indices)
    if saveAs != None:
        ig.plot(
            g,
            layout=ig.Layout(coords=g.vs['coordinates']),
            target="{saveAs}.png".format(saveAs=saveAs),
            vertex_size=15,
            edge_width=0.7,
            bbox=(1000,1000)
        )
    
    if(g.vs.select(class_eq=classmark, mark_eq=1).indices != [] and recursive):
        print(g.vs.select(class_eq=classmark, mark_eq=1))
        showgraph(g.subgraph(g.vs.select(mark_eq=1)), recursive=False)
    if changecolor:
        g.vs["color"] = oldcolor

""" plot the graph, color according to the attribute """
def ShowNumAttr(g, attr='cluster', continues=False, saveAs=None, noshow=False):
    oldcolor = g.vs["color"]

    def getcontcolor(pointvals):
        return px.colors.sample_colorscale(px.colors.sequential.Sunset, samplepoints=pointvals, colortype="rgb")
    if not continues: #discreet
        colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24 + ['#666666']
        g.vs["color"] = [colors[numattr] for numattr in g.vs[attr]]
    if continues:
        maxattr = max(g.vs[attr])
        minattr = min(g.vs[attr])
        pointvals = [(val-minattr)/(maxattr-minattr) for val in g.vs[attr]]
        g.vs["color"] = getcontcolor(pointvals)
    
    showgraph(g, recursive=False, changecolor=False, saveAs=saveAs, noshow=noshow)

    g.vs["color"] = oldcolor
    

""" partition the graph into two classes """
def genRandPartitioning(g, swapping=True, swappingprob=0.3):
    g.vs['class'] = RNG.integers(low=0, high=2, size=g.vcount())
    g = classcolor(g, showchanges=False)
    g.vs["shape"] = "square"
    #showgraph(g)
    if swapping:
        for v in g.vs:
            firstclass = RNG.choice([0,1])
            secondclass = 1-firstclass
            majoritythreshold = 1
            if (len(g.vs[g.neighbors(v.index, mode="ALL")].select(class_eq=firstclass))>len(g.vs[g.neighbors(v.index, mode="ALL")].select(class_eq=secondclass))+majoritythreshold) and RNG.random() < swappingprob:
                v["class"] = firstclass
            if (len(g.vs[g.neighbors(v.index, mode="ALL")].select(class_eq=secondclass))>len(g.vs[g.neighbors(v.index, mode="ALL")].select(class_eq=firstclass))+majoritythreshold) and RNG.random() < swappingprob:
                v["class"] = secondclass
    
    g = classcolor(g, showchanges=False)
    return g

""" partition the graph into two classes, with given probablities for verticies near the vorigin coordinates """
def genRandPartitioningOriginProb(g, originprob=1, sigmaDist=0.1, vorigins=None):
    global GLOBAL_DIMENSION
    if vorigins == None:
        vorigins = [{ "coordinates": [0 for _ in range(GLOBAL_DIMENSION)], "originprob": originprob, "sigmaDist": sigmaDist }]

    g.vs["class"] = [0 if any([d_euclidean(g,v,vorigin) < vorigin["sigmaDist"] and RNG.random()<vorigin["originprob"] for vorigin in vorigins])  else 1 for v in g.vs]
    g = classcolor(g, showchanges=False)
    g.vs["shape"] = "square"

    return g

def smoothoutPartitioning(g, iterations=1, classmark=0, majoritythreshold=1, both=False):
    firstclass = classmark
    secondclass = 1-firstclass
    for _ in range(iterations):
        for v in g.vs:
            if (len(g.vs[g.neighbors(v.index, mode="ALL")].select(class_eq=firstclass))>len(g.vs[g.neighbors(v.index, mode="ALL")].select(class_eq=secondclass))+majoritythreshold):
                v["class"] = firstclass
            if both:
                if (len(g.vs[g.neighbors(v.index, mode="ALL")].select(class_eq=secondclass))>len(g.vs[g.neighbors(v.index, mode="ALL")].select(class_eq=firstclass))+majoritythreshold):
                    v["class"] = secondclass
    g = classcolor(g, showchanges=False)
    return g

def smoothoutPartitioning2(g, iterations=1, classmark=0, onlyone=False, maxiterations=10):
    verticies = indextovertex(g, g.vs.select(class_eq=classmark).indices)
    def smoothfn():
        indlist = []
        for (vx,vy) in itertools.product(verticies, verticies):
            if vx.index == vy.index:
                continue
            sps = g.get_all_shortest_paths(vx, vy, mode="ALL")
            if len(sps[0]) == 0:
                continue
            if onlyone:
                sps = [sps[0]]
            for sp in sps:
                for v in sp:
                    if not g.vs[v]["class"] == classmark:
                        indlist.append(v)
                        #g.vs[v]["class"] = classmark
        for ind in list(set(indlist)):
            g.vs[ind]["class"] = classmark
            
    if iterations == 0:
        continueflag = True
        iterations = maxiterations
        while continueflag and iterations > 0:
            smoothfn()
            for (vx,vy) in itertools.product(verticies, verticies):
                sps = g.get_all_shortest_paths(vx, vy, mode="ALL")
                if onlyone:
                    sps = [sps[0]]
                for sp in sps:
                    if not all([g.vs[v]["class"] == classmark for v in sp]):
                        continueflag = True
                        break
                    continueflag = False
            iterations -= 1
    else:
        for _ in range(iterations):
            smoothfn()
    g = classcolor(g, showchanges=False)
    return g


""" calculates the sampled harmonic closeness centrality """
def sampledHarmonicClosenessCentrality(g, vertex, Landmark=0.1, distfn=GLOBAL_DISTANCE_FN):
    sample = set()
    if (type(Landmark) == float or Landmark == 1) and Landmark > 0 and Landmark <= 1:
        sample = set(RNG.choice(g.vs, size=math.floor(g.vcount()*Landmark), replace=False))
    if type(Landmark) == set:
        sample = Landmark
    sample = sample.difference(set([vertex]))

    def dfnn(v1, v2):
        res = distfn(g, v1, v2)
        if res == 0:
            return math.inf
        return res

    if len(sample) == 0:
        return 0
    return (1/len(sample)) * (sum([1/dfnn(vertex, v) for v in sample]))

""" return the n hop neighborhood of a vertex"""
def getvertexnh(g, vertex, n_hop=1):
    return set(g.vs.select(g.neighborhood(vertex.index, order=n_hop)))

""" calculate the sampled harmonic closeness centrality with a given landmark selection function """
def calculateSampledHarmonicClosenessCentrality(g, distfn=GLOBAL_DISTANCE_FN, samplepercantage=0.1, landmarkselectionFn=None):
    for v in g.vs:
        if landmarkselectionFn != None:
            v['hcc'] = sampledHarmonicClosenessCentrality(g, v, Landmark=landmarkselectionFn(g, v), distfn=distfn)
        else:
            v['hcc'] = sampledHarmonicClosenessCentrality(g, v, Landmark=samplepercantage, distfn=distfn)
    return g

""" calculates the degenracy of the graph"""
def calculateDegeneracy(g):
    g.vs['degeneracy'] = -1
    gcopy = copy.deepcopy(g)
    k = 0
    d_min = 0
    gcopy.vs['ind']=[v.index for v in gcopy.vs]
    while gcopy.vcount() > 0:
        vtx = gcopy.vs.select(_degree=d_min)
        if len(vtx) == 0:
            d_min = min(gcopy.degree())
            continue
        vtx = vtx[0]
        ind = vtx['ind']
        gcopy.delete_vertices(vtx)
        k = max(k, d_min)
        g.vs[ind]['degeneracy'] = k
    return g

""" calculates the components of a set of verticies in a graph """
def getComponents(g, verticies):
    verticies = verticies if isinstance(verticies, set) else set(verticies)
    components = []
    pq = queue.Queue()
    for v in verticies:
        pq.put(v)
    
    while not pq.empty():
        v = pq.get()
        if any([v in c for c in components]):
            continue
        c = set()
        c.add(v)
        for v2 in verticies.intersection(set(vertextoindex(g.neighborhood(v, order=1, mindist=1)))):                
            c.add(v2)
        components.append(c)

    # add all verticies to a component
    # merge components if they have a common vertex
    n = len(components)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if components[i].intersection(components[j]):
                uf.union(i, j)

    for i in range(n):
        for j in range(i + 1, n):
            if any([len(set(g.neighbors(v)).intersection(components[j])) for v in indextovertex(g, components[i])]):
                uf.union(i, j)

    merged_components = {}
    for i in range(n):
        root = uf.find(i)
        if root not in merged_components:
            merged_components[root] = components[i]
        else:
            merged_components[root].update(components[i])

    return list(merged_components.values())

""" calculates WCSS for a given partitioning """
def WCSS(g, As, distfn=d_euclidean):
    As = [indextovertex(g, Ai) for Ai in As]
    if distfn == d_step or distfn == d_geodesic:
        distfn = d_euclidean
    results = []
    for A in As:
        centeroid = {'coordinates': np.mean([v['coordinates'] for v in A], axis=0).tolist()}
        results.append(sum([distfn(g, v, centeroid) for v in A]))
    return sum(results)

""" calculates DaviesBouldinIndex score """
def DaviesBouldinIndexSi(g, As, q=1, distfn=d_euclidean):
    As = [indextovertex(g, Ai) for Ai in As]
    if distfn == d_step or distfn == d_geodesic:
        distfn = d_euclidean
    results = []
    for A in As:
        centeroid = {'coordinates': np.mean([v['coordinates'] for v in A], axis=0).tolist()}
        Si = (sum([distfn(g, v, centeroid)**q for v in A])*1/len(A))**(1/q)
        results.append(Si)
    return sum(results)

""" calculates Silhouette score"""
def Silhouette(g, As, distfn=d_step):
    As = [indextovertex(g, Ai) for Ai in As]
    if len(As) == 1:
        return 0
    results = []
    for A in As:
        for v in A:
            a = sum([distfn(g, v, v2) for v2 in A if v2 != v]) / (len(A)-1) if len(A) > 1 else 0
            b = min([sum([distfn(g, v, v2) for v2 in A2]) / len(A2) for A2 in As if A2 != A and len(A2) > 0])
            results.append((b-a)/max(a,b))
    return sum(results)

""" calculates the convexity of a set of verticies in a graph with one of the functions WCSS, DaviesBouldinIndex or Silhouette """
def calculateConvexity(g, A, distfn=d_step, fn=WCSS):
    As = getComponents(g, vertextoindex(A))
    As = [indextovertex(g, Ai) for Ai in As]
    return fn(g, As, distfn=distfn)


""" add cclusters to the graph """
def generateGraphWithOrigins(graph, ccluster, scaledfactor=None, dimension=None, precision=None):
    global GLOBAL_DIMENSION
    global PRECISION

    if dimension is None:
        dimension = GLOBAL_DIMENSION
    
    if precision is None:
        precision = PRECISION

    dias = []

    for dims in range(dimension):
        dias.append( abs(max([v["coordinates"][dims] for v in graph.vs])-min([v["coordinates"][dims] for v in graph.vs])) )


    if scaledfactor is None:
        scaledfactor = np.array(dias, dtype=np.float64)

    vorigins = []
    def createco():
        co = (RNG.random(size=(dimension))-0.5)*scaledfactor
        co = np.around(co, decimals=precision)
        co = { "coordinates": co.tolist(), "sigmaDist": 0.1*np.mean(scaledfactor), "originprob": 1 }
        return co

    for _ in range(ccluster):
        co = createco()
            
        while any([ d_euclidean(graph, co, originI) < 0.1*np.mean(scaledfactor) for originI in vorigins]):
            co = createco()

        vorigins.append(co)

    graph = genRandPartitioningOriginProb(graph, vorigins=vorigins)
    return graph