from threading import Thread
from queue import Queue
from time import sleep, time
import multiprocessing as mp
mp.set_start_method('fork')
import copy
from tqdm import tqdm
from typing import Union, Callable, List, Tuple, Any, Optional
import logging
import itertools
from graphfns import vertextoindex, indextovertex
from distances import d_step
import math
import numpy as np
import gc
import concurrent.futures
import random


def isIterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    return True

def flatten(l, dimension=1):
    if not isIterable(l):
        return l
    if dimension == 0:
        return l
    if dimension < -1:
        raise ValueError("Dimension must be greater than 0")
    if dimension == -1:
        while isIterable(l) and any(isIterable(sublist) for sublist in l):
            l = flatten(l, dimension=1)
        return l

    if dimension == 1:
        ret = []
        for sublist in l:
            if isIterable(sublist):
                ret += sublist
            else:
                ret.append(sublist)
        return ret
    if dimension >= 2:
        ret = []
        for sublist in l:
            if isIterable(sublist):
                ret += flatten(sublist, dimension=dimension-1)
            else:
                ret.append(sublist)
        return ret


def nopFn(arrEl):
    return arrEl

def alwaysTrueFn(el):
    return True

class SplitList:
    def __init__(self, arr, checkFn=alwaysTrueFn, parallel=False):
        if type(arr) == SplitList:
            arr = arr.returnList()
        self.arr = arr
        self.checkFn = checkFn
        self.parallel = parallel
        self.tfarr = self.__checkFn(self.checkFn, self.arr, cast=False)
        self.trueArr = [el[0] for el in self.tfarr if el[1]]
        self.falseArr = [el[0] for el in self.tfarr if not el[1]]
        self.index = -1
        self.checkAgainFlag = False

    def __parallelRun(self, arr, parallelFn):
        import multiprocessing as mp
        # if you need to run it in parallel, that probably means you have horrible functions and big data
        poolsize = mp.cpu_count()
        pool = mp.Pool(poolsize)
        chunks = [arr[i:i + poolsize] for i in range(0, len(arr), poolsize)]
        results = []
        for chunk in chunks:
            localresults = pool.map(parallelFn, chunk)
            print(chunk, localresults)
            for i,el in enumerate(chunk):
                results.append((el, localresults[i]))
        pool.close()
        return results
    
    def __checkFn(self, checkFn, arr, cast=True):
        res = []
        if not self.parallel:
            res = [(el, checkFn(el)) for el in arr]
        else:
            res = self.__parallelRun(arr, checkFn)
        return SplitList(res) if cast else res
    
    def __applyFnArrEl(self, fn, arr):
        if not self.parallel:
            return SplitList([fn(el) for el in arr])
        return SplitList([res[1] for res in self.__parallelRun(arr, fn)])
        
    def getCheckAgainFlag(self):
        return self.checkAgainFlag

    def renewcheck(self):
        self.tfarr = self.__checkFn(self.checkFn, self.arr)
        self.trueArr = SplitList([el[0] for el in self.tfarr if el[1]])
        self.falseArr = SplitList([el[0] for el in self.tfarr if not el[1]])
        self.checkAgainFlag = False
        return self
    
    def sort(self, key=None, reverse=False):
        self.arr = sorted(self.arr, key=key, reverse=reverse)
        self.checkAgainFlag = True
        return self
    
    def extend(self, arr):
        self.arr.extend(arr)
        self.checkAgainFlag = True
        return self

    def __str__(self):
        return f"<SplitList {str(self.arr)}>"

    def get(self, i):
        return self.arr[i]
    
    def __getitem__(self, key):
        arr = self.returnList()
        return arr[key]
    
    def __setitem__(self, key, value):
        self.arr = self.returnList()
        self.arr[key] = value
        self.arr = self.arr
        self.checkAgainFlag = True
        return self
    
    def __reversed__(self):
        self.arr = self.returnList()
        rev = reversed(self.arr)
        self.arr = list(rev)
        self.checkAgainFlag = True
        return self
        
    def __list__(self):
        self.arr = self.returnList()
        return self.arr

    def __next__(self):
        self.index += 1
        if self.index >= len(self.arr):
            self.index = -1
            raise StopIteration
        self.arr = self.returnList()
        return self.arr[self.index]

    def __len__(self):
        return len(self.arr)

    def getTrueArr(self):
        if self.checkAgainFlag:
            self.renewcheck()
        return self.trueArr
    def setTrueArr(self, arr):
        if len(arr) != len(self.trueArr):
            raise ValueError("Array length does not match")
        if type(arr) != SplitList:
            arr = SplitList(arr)
        self.trueArr = arr
        self.arr = self.returnList()
        self.checkAgainFlag = True
        return self
    def getFalseArr(self):
        if self.checkAgainFlag:
            self.renewcheck()
        return self.falseArr
    def setFalseArr(self, arr):
        if len(arr) != len(self.falseArr):
            raise ValueError("Array length does not match")
        if type(arr) != SplitList:
            arr = SplitList(arr)
        self.falseArr = arr
        self.arr = self.returnList()
        self.checkAgainFlag = True
        return self
    
    def delTrueArr(self):
        self.arr = list(self.falseArr)
        self.checkAgainFlag = True
        return self
    
    def delFalseArr(self):
        self.arr = list(self.trueArr)
        self.checkAgainFlag = True
        return self

    def pop(self, i):
        if i >= len(self.arr):
            raise IndexError("Index out of range")
        newarr = []
        ret = None
        ind = 0
        for obj in self.arr:
            if ind == i:
                ret = obj
            else:
                newarr.append(obj)
            ind += 1
        self.arr = newarr
        self.checkAgainFlag = True
        return ret
    
    def delInd(self, i):
        self.pop(self, i)
        return self
    
    def remEl(self, obj):
        newarr = []
        for el in self.arr:
            if el != obj:
                newarr.append(el)
        self.arr = newarr
        self.checkAgainFlag = True
        return self
    
    def appendWOCheck(self, obj):
        self.arr.append(obj)
        self.checkAgainFlag = True
        return self
    
    def append(self, obj):
        self.arr.append(obj)
        self.tfarr.appendWOCheck((obj, self.checkFn(obj)))
        if self.checkFn(obj):
            self.trueArr.appendWOCheck(obj)
        else:
            self.falseArr.appendWOCheck(obj)
        return self


    def applyToTrueArr(self, fn):
        self.trueArr = fn(self.trueArr)
        self.arr = self.returnList()
        return self
    
    def applyToFalseArr(self, fn):
        self.falseArr = fn(self.falseArr)
        self.arr = self.returnList()
        return self
    
    def applyToTrueArrEl(self, fn):
        self.trueArr = self.__applyFnArrEl(fn, self.trueArr)
        self.arr = self.returnList()
        return self
    
    def applyToFalseArrEl(self, fn):
        self.falseArr = self.__applyFnArrEl(fn, self.falseArr)
        self.arr = self.returnList()
        return self
    
    def combineagain(self):
        if self.checkAgainFlag:
            self.renewcheck()
        self.checkFn = alwaysTrueFn
        self.arr = [self.trueArr.pop(0) if tf else self.falseArr.pop(0) for el, tf in self.tfarr]
        self.renewcheck()
        return self
    
    def returnList(self):
        if self.checkAgainFlag:
            self.renewcheck()
        trueArr = copy.deepcopy(self.trueArr)
        falseArr = copy.deepcopy(self.falseArr)
        return [trueArr.pop(0) if tf else falseArr.pop(0) for el, tf in self.tfarr]
    
    def applynewSplit(self, checkFn):
        if self.checkAgainFlag:
            self.renewcheck()
        trueArr = copy.deepcopy(self.trueArr)
        falseArr = copy.deepcopy(self.falseArr)
        self.arr = [trueArr.pop(0) if tf else falseArr.pop(0) for el, tf in self.tfarr]
        self.checkFn = checkFn
        self.renewcheck()
        return self
    
    def apply(self, listObj):
        res=self
        for applyfn, paramfn in listObj:
            res = applyfn(res, paramfn)
        return res
    
    def retFunctions(self):
        def applyToTrueArr(slf, fn):
            slf.applyToTrueArr(fn)
            return slf
        def applyToFalseArr(slf, fn):
            slf.applyToFalseArr(fn)
            return slf
        def applyToTrueArrEl(slf, fn):
            slf.applyToTrueArrEl(fn)
            return slf
        def applyToFalseArrEl(slf, fn):
            slf.applyToFalseArrEl(fn)
            return slf
        def applynewSplit(slf, fn):
            slf.applynewSplit(fn)
            return slf
        return {
            "applyToTrueArr": applyToTrueArr,
            "applyToFalseArr": applyToFalseArr,
            "applyToTrueArrEl": applyToTrueArrEl,
            "applyToFalseArrEl": applyToFalseArrEl,
            "applynewSplit": applynewSplit,
        }


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                self.rank[root_y] += 1

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

RNG = np.random.default_rng(seed=102671797463975) 
def samplepairs(A, samplepercentage):
    if samplepercentage < 0 or samplepercentage > 1:
        raise ValueError("Sample percentage must be between 0 and 1")
    if samplepercentage == 0 or samplepercentage == 0.0:
        return []
    if samplepercentage == 1 or isclose(samplepercentage, 1.0):
        return list([(vx,vy) for vx,vy in itertools.product(A, A)])
    pairlist = []
    A = list(A)
    numberofsamples = math.floor(len(A)*samplepercentage)
    drawA = [(index, element) for index, element in enumerate(list(A) if samplepercentage < 0.5 else list(A)+list(A))]
    while numberofsamples > 0:
        v1 = RNG.choice(drawA)
        v2 = RNG.choice(drawA)
        if v1[1] == v2[1] or (v1[0]-v2[0]) % len(A) == 0:
            continue
        pairlist.append((v1[1],v2[1]))
        drawA = list(filter(lambda item: item[0]!=v1[0] and item[0]!=v2[0],drawA))
        numberofsamples -= 1
    return pairlist

def random_subset(s, length):
    out = set()

    sd = list(copy.deepcopy(s))

    if length > len(sd):
        raise ValueError("Length of subset is greater than length of set")
    
    for _ in range(length):
        out.add(sd.pop(RNG.integers(0, len(sd))))

    return list(out)

class DistanceCache:
    graph = None
    distances = {}
    defaultdistfn = None
    defaulthash = None

    def __init__(self, graph, defaultdistfn=None):
        self.graph = graph
        self.distances = {}
        self.defaultdistfn = defaultdistfn
        self.defaulthash = hash(defaultdistfn) if defaultdistfn is not None else None

    
    
    def join(self, other, ignore=True):
        if not ignore:
            if type(other) != DistanceCache:
                raise ValueError("Cannot join non DistanceCache object")
            if self.graph != other.graph:
                raise ValueError("Cannot join DistanceCache objects with different graphs")
            if self.defaultdistfn != other.defaultdistfn:
                raise ValueError("Cannot join DistanceCache objects with different defaultdistfn")
        self.distances.update(other.distances)

    def iv(self, i, g=None):
        if g is None:
            g = self.graph
        if i > g.vcount()-1:
            raise ValueError("DistanceCache: Index out of range" + str(i) + " " + str(g.vcount()-1))
        return indextovertex(g, i)
    
    def vi(self, v):
        return vertextoindex(v)
    
    def getvertexingraph(self, vertex, fromgraph=None, tograph=None):
        if tograph == None:
            tograph = self.graph
        if fromgraph == None:
            fromgraph = self.graph
        vertex = self.iv(vertex, g=fromgraph)
        vertex = self.vi(tograph.vs.select(id_eq=vertex["id"]))
        if len(vertex) < 1:
            raise ValueError("Vertex not found in graph")
        return vertex[0]

    def existsvertexingraph(self, vertex, fromgraph=None, tograph=None):
        vrt = self.getvertexingraph(vertex, fromgraph=fromgraph, tograph=tograph)
        return len(vrt) > 0
    
    def getverticiesidsingraph(self, verticies, fromgraph=None, tograph=None):
        if tograph == None:
            tograph = self.graph
        if fromgraph == None:
            fromgraph = self.graph
        verticies = [self.getvertexingraph(v, fromgraph=fromgraph, tograph=tograph) for v in verticies]
        return self.vi(verticies)
    
    def subset(self, vertexids):
        vertexids = self.vi(vertexids)
        subgraph = self.graph.subgraph(vertexids)
        newdc = DistanceCache(subgraph, defaultdistfn=self.defaultdistfn)
        mappingoldnew = {}
        mappingnewtoold = {}
        for v in vertexids:
            vert = self.getvertexingraph(v, fromgraph=self.graph, tograph=subgraph)
            mappingoldnew[v] = vert
            mappingnewtoold[vert] = v
        for vx,vy in itertools.product(vertexids, vertexids):
            for distfnhash in self.distances:
                newdc.setdistance(mappingoldnew[vx], mappingoldnew[vy], self.getDistance(vx, vy), distfnhash=distfnhash)
        return newdc, mappingoldnew, mappingnewtoold




    def precomputeDistances(self, verticies=None, distfn=None):
        if verticies is None:
            verticies = self.graph.vs

        if distfn is None:
            distfn = self.defaultdistfn
        
        distfnhash = hash(distfn)
        if not distfnhash in self.distances:
            self.distances[distfnhash] = {}

        verticies = self.vi(verticies)

        for v in verticies:
            self.distances[distfnhash][v] = { }
            
        for v, v2 in tqdm(itertools.product(verticies, verticies), total=len(verticies)**2):
            if v2 in self.distances[distfnhash][v] or v in self.distances[distfnhash][v2]:
                continue
            v1v = self.iv(v)
            v2v = self.iv(v2)
            self.distances[distfnhash][v][v2] = distfn(self.graph, v1v, v2v)
            self.distances[distfnhash][v2][v] = self.distances[distfnhash][v][v2]

    def setdistance(self,v1,v2,distance, distfnhash=None):
        if distfnhash is None:
            distfn = self.defaultdistfn
            distfnhash = hash(distfn)

        if not distfnhash in self.distances:
            self.distances[distfnhash] = {}
        v1 = self.vi(v1)
        v2 = self.vi(v2)
        if not v1 in self.distances[distfnhash]:
            self.distances[distfnhash][v1] = {}
        if not v2 in self.distances[distfnhash]:
            self.distances[distfnhash][v2] = {}
        self.distances[distfnhash][v1][v2] = distance
        self.distances[distfnhash][v2][v1] = distance



    def getDistFnHash(self, distfn):
        return hash(distfn)

    def setDefaultDistFn(self, distfn):
        self.defaultdistfn = distfn
        self.defaulthash = hash(distfn)

    def clearDefaultDistFn(self):
        self.defaultdistfn = None
        self.defaulthash = None

    def getDistancePrecomputed(self, v1, v2):
        return self.distances[self.defaulthash][v1][v2]


    def getDistance(self, v1, v2, distfnhash=None, distfn=None, precomputeIfNotExists=False, computeIfNotExists=True):
        if distfnhash is None and distfn is None:
            if self.defaultdistfn is None:
                raise ValueError("Must pass either distfnhash or distfn")
            distfn = self.defaultdistfn
            distfnhash = self.getDistFnHash(distfn)

        if distfnhash is None and distfn is not None:
            distfnhash = self.getDistFnHash(distfn)
        
        if not distfnhash in self.distances:
            self.distances[distfnhash] = {}

        if precomputeIfNotExists:
            if distfn is None:
                raise ValueError("Must pass distfn or have default distfn if precomputeIfNotExists is True")
            self.precomputeDistances(distfn=distfn)
        
        if not v1 in self.distances[distfnhash] or not v2 in self.distances[distfnhash][v1]:
            if not computeIfNotExists:
                raise ValueError("Distance not precomputed, nor should be computed")
            if distfn is None:
                raise ValueError("Must pass distfn or have default distfn if computeIfNotExists is True")
            v1v = self.iv(v1)
            v2v = self.iv(v2)

            dist = distfn(self.graph, v1v, v2v)

            if not v1 in self.distances[distfnhash]:
                self.distances[distfnhash][v1] = {}
            if not v2 in self.distances[distfnhash]:
                self.distances[distfnhash][v2] = {}

            self.distances[distfnhash][v1][v2] = dist
            self.distances[distfnhash][v2][v1] = dist
            return dist


        return self.distances[distfnhash][v1][v2]

class AreaCache:
    graph = None
    NHareas = {}
    NHareasSE = {}
    
    DeltaAreas = {}
    Zsets = {}
    defaultdistfn = None
    distancecache = None
    precomputedDistances = False
    precomputedZsets = {}
    precomputedAT = {}
    precomputedE = {}
    precomputedSP = {}
    longestpathlen = 0

    subgraphs = {}
    subgraphsSP = {}

    def __init__(self, graph, defaultdistfn=None, distancecache=None, pcdistances=False):
        self.graph = graph
        if distancecache is None:
            self.distancecache = DistanceCache(graph, defaultdistfn=defaultdistfn)
        else:
            self.distancecache = distancecache
        self.NHareas = {}
        self.NHareasSE = {}

        self.subgraphs = {}
        self.subgraphsSP = {}

        self.defaultdistfn = hash(defaultdistfn) if defaultdistfn is not None else None
        self.DeltaAreas = {}
        self.precomputedDistances = False
        self.precomputedZsets = {}
        self.precomputedAT = {}
        self.precomputedE = {}
        self.precomputedSP = {}
        self.longestpathlen = 0
        self.parallelFnEx = parallelFun(minusX=1)
        if pcdistances:
            self.precomputeDistances()
            self.precomputeLongestSP()
            self.precomputeNHAreas()
            self.precomputeZsets(epsilon=0.0)
            maxlen = self.getLongestPossiblePath()
            for l in range(1, maxlen+1):
                self.precomputeNHAreas(theta=float(l))


    def join(self, other, ignore=True):
        if other is None:
            return self
        
        if type(other) != AreaCache:
            return self
        
        if not ignore:
            if type(other) != AreaCache:
                raise ValueError("Cannot join non AreaCache object")
            if self.graph != other.graph:
                raise ValueError("Cannot join AreaCache objects with different graphs")
            if self.distancecache != other.distancecache:
                raise ValueError("Cannot join AreaCache objects with different distancecache")
            if self.defaultdistfn != other.defaultdistfn:
                raise ValueError("Cannot join AreaCache objects with different defaultdistfn")
            if self.precomputedDistances != other.precomputedDistances:
                raise ValueError("Cannot join AreaCache objects with different precomputedDistances")
        
        self.NHareas.update(other.NHareas)
        self.NHareasSE.update(other.NHareasSE)
        self.DeltaAreas.update(other.DeltaAreas)
        self.Zsets.update(other.Zsets)
        self.precomputedZsets.update(other.precomputedZsets)
        self.precomputedAT.update(other.precomputedAT)
        self.precomputedE.update(other.precomputedE)
        self.precomputedSP.update(other.precomputedSP)
        self.longestpathlen = max(self.longestpathlen, other.longestpathlen)
        self.subgraphs.update(other.subgraphs)
        self.subgraphsSP.update(other.subgraphsSP)
        self.distancecache.join(other.distancecache)


    def iv(self, i, g=None):
        if g is None:
            g = self.graph
        if i > g.vcount()-1:
            raise ValueError("Index out of range" + str(i) + " " + str(g.vcount()-1))
        return indextovertex(g, i)
    
    def vi(self, v):
        return vertextoindex(v)



    def getLongestPossiblePath(self, setA=None):
        if setA is None:
            return self.longestpathlen
        setA = self.vi(setA)
        locallongestpathlen = 0
        for vx,vy in itertools.product(setA, setA):
            if vx == vy:
                continue
            paths = self.computeSP(vx, vy)
            if len(paths) != 0:
                locallongestpathlen = max(locallongestpathlen, max([len(sp) for sp in paths]))
        return locallongestpathlen


    def setDefaultDistFn(self, distfn):
        self.defaultdistfn = hash(distfn)
        self.distancecache.setDefaultDistFn(distfn)
    
    def clearDefaultDistFn(self):
        self.defaultdistfn = None
        self.distancecache.clearDefaultDistFn()

    def precomputeDistances(self, verticies=None):
        if verticies is None:
            verticies = self.graph.vs
        self.distancecache.precomputeDistances(verticies=verticies)
        self.precomputedDistances = True

    def __distfn(self, v1, v2, distfn=None):
        if self.precomputedDistances:
            return self.distancecache.getDistancePrecomputed(v1, v2)
        
        if distfn is not None:
            return self.distancecache.getDistance(v1, v2, distfn=distfn)
        return self.distancecache.getDistance(v1, v2)

        
    def getDisCache(self):
        return self.distancecache

    def dnhfactory(self, vertex, delta, distfn=None):
        return self.dnhfactorySE(vertex, delta, distfn=distfn)
        # vertex = self.vi(vertex)
        # def rtfn(v):
        #     v = self.vi(v)
        #     return self.__distfn(vertex, v) < delta
        # return rtfn
    
    def dnhfactorySE(self, vertex, delta, distfn=None):
        vertex = self.vi(vertex)
        def rtfn(v):
            v = self.vi(v)
            return self.__distfn(vertex, v) <= delta
        return rtfn
    
    def _mequals(self, f1, f2):
        return math.isclose(f1, f2, rel_tol=1e-09, abs_tol=0.0)
    
    def getDistance(self, v1, v2):
        return self.distancecache.getDistance(v1, v2)

    def triangleDeltaFactory(self, vx, vy, epsilon=0.0, distfn=None):
        vx = self.vi(vx)
        vy = self.vi(vy)
        def rtfn(v):
            v = self.vi(v)
            if epsilon == 0.0:
                f1 = self.__distfn(vx, v) + self.__distfn(v, vy)
                f2 = self.__distfn(vx, vy)
                return self._mequals(f1, f2)
            return self.__distfn(vx, v) + self.__distfn(v, vy) <= self.__distfn(vx, vy) + epsilon
        return rtfn
    
    def Delta(self, x, y, epsilon=0.0, distfn=None):
        return set(self.vi(self.graph.vs.select(self.triangleDeltaFactory(x, y, epsilon=epsilon))))

    def precomputeNHAreas(self, theta=1.0):
        for el in tqdm(self.vi(self.graph.vs), total=len(self.graph.vs)):
            if not el in self.NHareas:
                self.NHareas[el] = {}
            if not el in self.NHareasSE:
                self.NHareasSE[el] = {}
            if not theta in self.NHareas[el]:
                self.computeNHAreas(el, theta)
            if not theta in self.NHareasSE[el]:
                self.computeNHSEAreas(el, theta)

    def computeNHAreas(self, el, theta, distfn=None):
        if not el in self.NHareas:
            self.NHareas[el] = {}
        if not theta in self.NHareas[el]:
            self.NHareas[el][theta] = set(self.vi(self.graph.vs.select(self.dnhfactory(el, theta))))

        return self.NHareas[el][theta]
    
    def setNHAreas(self, el, theta, nhareas):
        if not el in self.NHareas:
            self.NHareas[el] = {}
        self.NHareas[el][theta] = nhareas
        return self
    
    def _getNHAreas(self):
        return self.NHareas
    
    def computeNHSEAreas(self, el, theta, distfn=None):
        if not el in self.NHareasSE:
            self.NHareasSE[el] = {}
        if not theta in self.NHareasSE[el]:
            self.NHareasSE[el][theta] = set(self.vi(self.graph.vs.select(self.dnhfactorySE(el, theta))))

        return self.NHareasSE[el][theta]
    
    def setNHSEAreas(self, el, theta, nhareas):
        if not el in self.NHareasSE:
            self.NHareasSE[el] = {}
        self.NHareasSE[el][theta] = nhareas
        return self
    
    def _getNHSEAreas(self):
        return self.NHareasSE
    
    def computeDeltaAreas(self, vx, vy, epsilon=0.0, distfn=None):
        if not vx in self.DeltaAreas:
            self.DeltaAreas[vx] = {}
        if not vy in self.DeltaAreas[vx]:
            self.DeltaAreas[vx][vy] = self.Delta(vx, vy, epsilon=epsilon)

        return self.DeltaAreas[vx][vy]
    
    def setDeltaAreas(self, vx, vy, deltaareas):
        if not vx in self.DeltaAreas:
            self.DeltaAreas[vx] = {}
        self.DeltaAreas[vx][vy] = deltaareas
        return self
    
    def _getDeltaAreas(self):
        return self.DeltaAreas
    
    def computeSP(self, vx, vy):
        if not vx in self.precomputedSP:
            self.precomputedSP[vx] = {}
        if not vy in self.precomputedSP:
            self.precomputedSP[vy] = {}
        if not vy in self.precomputedSP[vx]:
            self.precomputedSP[vx][vy] = self.graph.get_all_shortest_paths(self.iv(vx), self.iv(vy), weights=None)
            if any([len(sp) > self.longestpathlen for sp in self.precomputedSP[vx][vy]]):
                self.longestpathlen = max(self.longestpathlen, max([len(sp) for sp in self.precomputedSP[vx][vy]]))
        if not vx in self.precomputedSP[vy]:
            self.precomputedSP[vy][vx] =  [list(reversed(spel)) for spel in self.precomputedSP[vx][vy]]
        return self.precomputedSP[vx][vy]
    
    def setSP(self, vx, vy, sp):
        if not vx in self.precomputedSP:
            self.precomputedSP[vx] = {}
        if not vy in self.precomputedSP:
            self.precomputedSP[vy] = {}
        self.precomputedSP[vx][vy] = sp
        self.precomputedSP[vy][vx] = [list(reversed(spel)) for spel in sp]
        return self
    
    def _getSP(self):
        return self.precomputedSP
    
    def _precomputeSPParalle(self, mis):
        v = self.iv(mis)
        vsis = self.vi(self.graph.vs)
        sps = self.graph.get_all_shortest_paths(v, to=vsis, weights=None)
        spls = {}
        for sp in sps:
            endv = sp[-1]
            if not endv in spls:
                spls[endv] = []
            spls[endv].append(sp) 
        mx = max([len(sp) for sp in sps])
        retobj = (mis, spls, mx)
        return retobj

    def precomputeSPPool(self):
        vertlist = self.vi(self.graph.vs)
        results = self.parallelFnEx.run(fn=self._precomputeSPParalle, listit=vertlist)
        for res in results:
            i, spls, longestpathlen = res
            if i is None:
                continue
            for key in spls:
                self.setSP(i, key, spls[key])
            self.longestpathlen = max(self.longestpathlen, longestpathlen)

    def _precomputelongestSPParallel(self, mis=None):
        if mis is None:
            return 0
        vsis = self.vi(self.graph.vs)
        v = self.iv(mis)
        sps = self.graph.get_shortest_paths(v, to=vsis, weights=None)
        sps = [len(sp) for sp in sps]
        return max(sps)


    def precomputeLongestSP(self):
        vertlist = vertextoindex(self.graph.vs)
        results = self.parallelFnEx.run(fn=self._precomputelongestSPParallel, listit=vertlist)
        maxlen = max(results)
        self.longestpathlen = max(self.longestpathlen, maxlen)
        return self.longestpathlen


    def precomputeSP(self, pool=False):
        if pool:
            self.precomputeSPPool()
            return
        for vx in tqdm(self.vi(self.graph.vs), total=len(self.graph.vs)):
            sps = self.graph.get_all_shortest_paths(self.iv(vx), to=self.vi(self.graph.vs), weights=None)
            spls = {}
            for sp in sps:
                endv = sp[-1]
                if not endv in spls:
                    spls[endv] = []
                spls[endv].append(sp) 

            for key in spls:
                self.setSP(vx, key, spls[key])

            self.longestpathlen = max(self.longestpathlen, max([len(sp) for sp in sps]))

    def setZsetTrueFalse(self, epsilon, trf=True):
        self.precomputedZsets[epsilon] = trf

    def precomputeZsets(self, epsilon):
        verticies = self.vi(self.graph.vs)
        for el in verticies:
            self.Zsets[el] = {}
            for nhel in verticies:
                self.Zsets[el][nhel] = {}

        for el, nhel in tqdm(itertools.product(verticies, verticies), total=len(verticies)**2):
            dist = self.__distfn(el, nhel)
            nhDistEl = self.computeNHAreas(el, dist)
            nhDistNhel = self.computeNHAreas(nhel, dist)
            precomputeDeltaAreas = self.computeDeltaAreas(el, nhel, epsilon=epsilon)
            zset = (nhDistEl.intersection(nhDistNhel)).intersection(precomputeDeltaAreas)
            self.Zsets[el][nhel][epsilon] = zset
            self.Zsets[nhel][el][epsilon] = zset
        self.precomputedZsets[epsilon] = True

    
    def computeZsets(self, el, nhel, epsilon):
        if epsilon in self.precomputedZsets and self.precomputedZsets[epsilon]:
            return self.Zsets[el][nhel][epsilon]
        
        if not el in self.Zsets:
            self.Zsets[el] = {}
        if not nhel in self.Zsets[el]:
            self.Zsets[el][nhel] = {}
        if not nhel in self.Zsets:
            self.Zsets[nhel] = {}
        if not el in self.Zsets[nhel]:
            self.Zsets[nhel][el] = {}

        if not epsilon in self.Zsets[el][nhel] or not epsilon in self.Zsets[nhel][el]:
            dist = self.__distfn(el, nhel)
            nhDistEl = self.computeNHAreas(el, dist)
            nhDistNhel = self.computeNHAreas(nhel, dist)
            precomputeDeltaAreas = self.computeDeltaAreas(el, nhel, epsilon=epsilon)
            zset = (nhDistEl.intersection(nhDistNhel)).intersection(precomputeDeltaAreas)
            self.Zsets[el][nhel][epsilon] = zset
            self.Zsets[nhel][el][epsilon] = zset
        return self.Zsets[el][nhel][epsilon]
    
    def setZsets(self, el, nhel, epsilon, zset):
        if not el in self.Zsets:
            self.Zsets[el] = {}
        if not nhel in self.Zsets[el]:
            self.Zsets[el][nhel] = {}
        if not nhel in self.Zsets:
            self.Zsets[nhel] = {}
        if not el in self.Zsets[nhel]:
            self.Zsets[nhel][el] = {}
        self.Zsets[el][nhel][epsilon] = zset
        self.Zsets[nhel][el][epsilon] = zset
        return self
    
    def _getZsets(self):
        return self.Zsets
    
    def setToKey(self, s):
        return str(sorted(list(self.vi(s))))
    
    def getInducedSubgraph(self, P):
        P = self.vi(P)
        Pkey = self.setToKey(P)
        if not Pkey in self.subgraphs:
            self.subgraphs[Pkey] = self.graph.subgraph(self.iv(P))
        return self.subgraphs[Pkey]
    
    def getSPinSubgraph(self, vx, vy, P):
        P = self.vi(P)
        Pkey = self.setToKey(P)
        if not Pkey in self.subgraphsSP:
            self.subgraphsSP[Pkey] = {}
        if not vx in self.subgraphsSP[Pkey]:
            self.subgraphsSP[Pkey][vx] = {}
        if not vy in self.subgraphsSP[Pkey]:
            self.subgraphsSP[Pkey][vy] = {}
        if not vy in self.subgraphsSP[Pkey][vx] or not vx in self.subgraphsSP[Pkey][vy]:
            subgraph = self.getInducedSubgraph(P)
            sgvx = self.getVertexinSubgraph(subgraph, vx)
            sgvy = self.getVertexinSubgraph(subgraph, vy)
            if len(sgvx) == 0 or len(sgvy) == 0:
                self.subgraphsSP[Pkey][vx][vy] = []
                self.subgraphsSP[Pkey][vy][vx] = []
                return self.subgraphsSP[Pkey][vx][vy]
            
            if len(sgvx) > 1 or len(sgvy) > 1:
                raise("WARNING: More than one vertex found in subgraph ", sgvx, sgvy)
                         
            sgvx = sgvx[0]
            sgvy = sgvy[0]
            sps = subgraph.get_shortest_paths(sgvx, sgvy, output="vpath")
            sps = [[self.vi(self.getVertexinGraph(self.iv(vertex, g=subgraph))) for vertex in sp] for sp in sps]

            self.subgraphsSP[Pkey][vx][vy] = sps
            self.subgraphsSP[Pkey][vy][vx] = sps
        
        return self.subgraphsSP[Pkey][vx][vy]


    def getVertexinSubgraph(self, subgraph, v):
        vertex = self.iv(v)
        return subgraph.vs.select(id_eq=vertex["id"])
    
    def getVertexinGraph(self, vertex):
        return self.graph.vs.select(id_eq=vertex["id"])
    

    def ec_pathnotinSet(self, path, setP):
        setP = self.vi(setP)
        return any([self.vi(p) not in setP for p in path])


    def hasSPinSetP2(self, vx, vy, setP, distance): # does a sp up to  length distance exist between vx and vy in setP
        sps = self.getSPinSubgraph(vx, vy, setP)

        sps = list(filter(lambda sp: len(sp) > 0, sps)) #ignore empty paths

        if len(sps) < 1:
            return False
            
        if float(len(sps[0])) > distance:
            return False
            
        if any([not self.ec_pathnotinSet(sp, setP) for sp in sps]):
            return True

        return False
    
    def hasSPinSetP(self, vx, vy, setP, distance):
        sps = self.computeSP(vx, vy)
        sps = list(filter(lambda sp: not self.ec_pathnotinSet(sp, setP), sps)) 
        if len(sps) < 1:
            return False
        if float(len(sps[0])) > distance:
            return False
        return True
        

    
    def ec_Vin(self, p, setP, Pd, delta=1.0):
        neighborhood = self.computeNHSEAreas(p, delta)
        setPd = set(self.vi(Pd))
        neighborhood = neighborhood.intersection(setPd)
        neighborhood = list(filter(lambda v: self.hasSPinSetP(p, v, setP, delta), list(neighborhood)))
        return set(neighborhood)
    
    def ec_Vin_O(self, p, setP, Pd, delta=1.0):
        neighborhood = self.computeNHSEAreas(p, delta)
        setPd = set(self.vi(setP))
        neighborhood = neighborhood.intersection(setPd)
        neighborhood = list(filter(lambda v: self.hasSPinSetP(p, v, setP, delta), list(neighborhood)))
        return set(neighborhood)

    def ec_Vall(self, p, P, Pd, delta=1.0):
        return self.computeNHSEAreas(p, delta).intersection(set(self.vi(Pd)))

    def ec_AT(self, P, Pd, delta=1.0):
        return sum([len(self.ec_Vall(p, P, Pd, delta=delta)) for p in Pd]) #filterpaths=False 

    
    def ec_AT_O(self, P, delta=1.0):
        argdelta=delta
        Pkey = self.setToKey(P)
        if not argdelta in self.precomputedAT:
            self.precomputedAT[argdelta] = {}
        if not Pkey in self.precomputedAT[argdelta]:
            self.precomputedAT[argdelta][Pkey] = sum([len(self.ec_Vall(p, P, delta=argdelta)) for p in P]) #filterpaths=False 
        return self.precomputedAT[argdelta][Pkey]
    
    def ec_f(self, p, P, Pd, delta=1.0):
        argdelta=delta
        ecvin = self.ec_Vin(p, P, Pd, delta=argdelta)
        return len(ecvin) / self.ec_AT(P, Pd, delta=argdelta)

    def ec_E(self, P, Pd, delta=1.0):
        argdelta=delta
        ecfres = [self.ec_f(p,P, Pd, delta=argdelta) for p in P]
        ecfres = list(filter(lambda x: x != 0.0 and x!=1.0, ecfres))
        try:
            ecfres = [res*math.log(res, math.e) for res in ecfres]
        except Exception as e:
            print("ecfres", ecfres)
            raise e
        return -sum(ecfres)
    
    def ec_E_O(self, P, delta=1.0):
        argdelta=delta
        Pkey = self.setToKey(P)
        if not argdelta in self.precomputedE:
            self.precomputedE[argdelta] = {}
        if not Pkey in self.precomputedE[argdelta]:
            ecfres = [self.ec_f(p,P, P, delta=argdelta) for p in P]
            ecfres = list(filter(lambda x: x != 0.0 and x!=1.0, ecfres))
            try:
                ecfres = [res*math.log(res, math.e) for res in ecfres]
            except Exception as e:
                print("ecfres", ecfres)
                raise e
            self.precomputedE[argdelta][Pkey] = -sum(ecfres)
        return self.precomputedE[argdelta][Pkey]
    
    def ec_Emax(self, Pd):
        return -math.log(1/len(Pd), math.e)
    
    def EntropyConvexity(self, P, delta=1.0, samplepercentage=1):
        if samplepercentage == 1:
            Pd = P
        else:
            sp = int(samplepercentage*len(P))
            if sp < 1:
                sp = 1
            Pd = random_subset(P, sp)
    
        argdelta = delta
        if len(Pd) == 1:
            return self.ec_E(P, Pd, delta=argdelta)
        ecE = self.ec_E(P, Pd, delta=argdelta)
        ecEmax = self.ec_Emax(Pd)
        return ecE / ecEmax
    


    def pathweight(self, path, A):
        if len(path) == 0:
            return math.inf
        path = self.vi(path)
        return sum([1 for p in path if p not in self.vi(A)])
    
    def gconfm1A(self, A, samplepercentage=1):
        # for all the paths we compute the average number of elements outside A
        if len(A) == 1 or len(A) == 0:
            return 0.0
        summed = 0
        countersum = 0

        pairslist = samplepairs(A, samplepercentage)

        for vx,vy in pairslist:
            if vx == vy:
                continue
            paths = self.computeSP(self.vi(vx), self.vi(vy))
            if len(paths) != 0:
                countersum += len(paths[0])
            else:
                continue

            paths = [self.pathweight(path, A) for path in paths]
            summed += np.mean(paths)
        return 1- (summed / countersum)
    
    def gconfm1Aminmax(self, A, usemax=False, samplepercentage=1):
        # for all the paths we compute the minimum number of elements outside A
        if len(A) == 1 or len(A) == 0:
            return 0.0
        summed = 0
        countersum = 0

        pairslist = samplepairs(A, samplepercentage)
        for vx,vy in pairslist:
            if vx == vy:
                continue
            paths = self.computeSP(self.vi(vx), self.vi(vy))
            if len(paths) != 0:
                countersum += len(paths[0])
            else:
                continue

            paths = [self.pathweight(path, A) for path in paths]
            if usemax:
                summed += max(paths)
            else:
                summed += min(paths)
        return 1- (summed / countersum)
    
    def gconfm1B(self, A, samplepercentage=1):
        # we count the number of paths that have at least one element outside A
        if len(A) == 1 or len(A) == 0:
            return 0.0
        summed = 0

        pairslist = samplepairs(A, samplepercentage)
        for vx,vy in pairslist:
            if vx == vy:
                continue
            paths = self.computeSP(self.vi(vx), self.vi(vy))
            if len(paths) == 0:
                continue
            pathsdenominator = [self.pathweight(path, A) for path in paths]
            pathsnominator = list(filter(lambda p: p>0, pathsdenominator)) #all paths that are not 0 / have an element outside A
            
            summed += len(pathsnominator)/len(pathsdenominator)
        return 1- (summed / (len(pairslist)))

def createTaskQueue():
    def threadedDoQueue(que, msc=100):
        while True:
            if not que.empty():
                que.get()()
            else:
                sleep(msc/1000)
    que = Queue()
    thr = Thread(target=threadedDoQueue, args=(que,))
    thr.start()
    return que, thr


def prlfn(arg):
    arg, i, parallelFn = arg
    i=i+1
    print("## Starting Worker {i} ##\n".format(i=i))
    start = time()
    result = parallelFn(arg)
    duration = time()-start
    print("## Ending Worker {i} time: {t} ##\n".format(i=i, t=duration))
    return (result, duration)

def prlfn2(arg):
    arg, i, parallelFn = arg
    result = parallelFn(arg)
    sleep(1)
    return (result, 0)

class TimeMeasure:
    def __init__(self, name):
        self.name = name
        self.start = time()
    def end(self):
        print(self.name, "took", time()-self.start, "seconds")
        return time()-self.start
    def equite(self):
        return time()-self.start
    def renew(self):
        t = time()-self.start
        self.start = time()
        return t

class MyAsyncResult():
    def __init__(self, res):
        self.res = res
        self.read = False
        self.starttime = time()
    def ready(self):
        if self.res is None:
            return False
        return self.res.ready()
    def get(self):
        self.read = True
        if self.res is None:
            return False, 0
        res = self.res.get()
        return res, time() - self.starttime
    def wasread(self):
        return self.read
    def isready(self):
        if self.res is None:
            return False
        return self.res.ready() and not self.read
    def rem(self):
        del self.res
        self.res = None
        gc.collect()
    def istimeout(self, timeout):
        if timeout is None:
            return False
        return time() - self.starttime > timeout

def createMPPool(parallelFn, iteratorargFn, trials=10, minusX=1, to=60*60*1.5, checkinterval=0.1, justresults=False):
    pfc = parallelFun(minusX=minusX, timeout=to, checkinterval=checkinterval, justresults=justresults)
    return pfc.createMPPool(iteratorargFn, parallelFn, trials=trials)

def processaftereachother(parallelFn, iteratorargFn, trials=10, minusX=1, to=60*60*1.5, checkinterval=0.1, justresults=False):
    pfc = parallelFun(timeout=to, checkinterval=checkinterval, justresults=justresults, cpusused=1)
    return pfc.createMPPool(iteratorargFn, parallelFn, trials=trials)
            
class parallelFun:
    minusX = 1
    fn = None
    listit = None
    length = 0
    nocpus = mp.cpu_count()
    nochunks = nocpus-minusX
    justresults = True
    timeout = None
    checkinterval = 0.1
    results = []
    times = []
    totaltime = 0

    def __init__(self, fn=None, listit=None, minusX=1, justresults=True, timeout=60*60*1.5, checkinterval=0.1, cpusused=None, nochunks=None):
        if cpusused is None:
            self.minusX = minusX
        else:
            self.minusX = mp.cpu_count() - cpusused
        if fn is not None:
            self.fn = fn
        if listit is not None:
            random.shuffle(listit)
            self.listit = listit
            self.length = len(listit)

        self.nocpus = mp.cpu_count()

        if nochunks is not None:
            self.nochunks = nochunks
        else:
            self.nochunks = self.nocpus-self.minusX

        self.justresults = justresults
        self.timeout = timeout
        self.checkinterval = checkinterval

    def run(self, fn=None, listit=None, justresults=True):
        if fn is not None:
            self.fn = fn
        if listit is not None:
            random.shuffle(listit)
            self.listit = listit
            self.length = len(listit)
        results = self.createMPPool(justresults=False)
        self.results = flatten([res[0] for res in results], dimension=1)
        self.times = flatten([res[1] for res in results], dimension=1)
        self.totaltime = np.mean(flatten([res[2] for res in results], dimension=1))
        if justresults:
            return self.results
        return self.results, self.times, self.totaltime

    def getresults(self):
        return self.results

    def gettimes(self):
        return self.times
    
    def gettotaltime(self):
        return self.totaltime
    
    def createMPPool(self, iteratorargFn=None, parallelFn=None, trials=None, minusX=None, to=None, checkinterval=None, justresults=None):
        starttime = time()
        if trials is None:
            trials = self.nochunks
        if minusX is None:
            minusX = self.minusX
        if justresults is None:
            justresults = self.justresults
        if checkinterval is None:
            checkinterval = self.checkinterval
        if to is None:
            to = self.timeout

        if iteratorargFn is None:
            iteratorargFn = self._foriterator
        if parallelFn is None:
            parallelFn = self._parallelFn
        

        numberOfProcesses = min(self.nocpus-self.minusX, trials)
        i = 0
        asyncresults = []
        results = []
        upto = min(numberOfProcesses, trials)
        with mp.Pool(processes=numberOfProcesses, maxtasksperchild=10) as pool, tqdm(total=trials, desc="Processing") as pbar:
            while i < trials:
                running = len(asyncresults)

                for k in range(upto):
                    if i+k+running >= trials:
                        break
                    asyncres = MyAsyncResult(pool.apply_async(parallelFn, args=(iteratorargFn(i+k+running),)))
                    asyncresults.append(asyncres)
                upto = 0
                sleep(checkinterval)
                for res in asyncresults:
                    if res.isready():
                        pbar.update(1)
                        results.append(res.get())
                        i += 1
                        upto += 1
                        res.rem()
                        print(i, upto, trials)
                    if res.istimeout(to):
                        res.rem()
                        upto += 1
                        i += 1
                        
                asyncresults = list(filter(lambda x: not x.wasread(), asyncresults))
            pool.close()
            pool.join()

        results = list(filter(lambda x: x is not None, results))
        te = time() - starttime
        results = [(res[0], res[1], te)  for res in results]

        if justresults:
            results = [res[0] for res in results]
        return results

    def _foriterator(self, j):
        
        n = (self.length // self.nochunks)
        n = n if n > 0 else 1
        iterl = [self.listit[i:i + n] for i in range(0, self.length, n)]
        print(j, len(iterl), iterl[j] if j < len(iterl) else None)

        if j >= self.nochunks or j < 0 or j >= len(iterl):
            return None

        return iterl[j]

    def _parallelFn(self, listobj):
        if listobj is None:
            return []

        return [self.fn(el) for el in listobj]

def ifprint(arg, toprint):
    if arg:
        print(toprint)

def printD(msg, debug=False):
    if debug:
        print(msg)
LOG = None
def getLogger(name):
    global LOG
    if LOG is not None:
        return LOG
    LOG = logging.getLogger(name)
    return LOG
def printdebug(_debug=False, _name="debug", *args, **kwargs):
    log = getLogger(_name)
    if _debug:
        log.debug(*args, **kwargs)
def crdbg(_debug=True, _name="debug", logging=False):
    def rtfn(*args, **kwargs):
        log = getLogger(_name)
        if _debug:
            if logging:
                log.debug(*args, **kwargs)
            else:
                print(*args, **kwargs)
    return rtfn