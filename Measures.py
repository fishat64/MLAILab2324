from lib import indextovertex, samplepairs, AreaCache
from OldExtensionalWeaklyConvexHull import Delta
from distances import d_step

def convSanity1(A, convexhullA):
    if len(convexhullA) == 0:
        return 0
    return len(A) / len(convexhullA)

def convSanity2(g, A, convexhullA, epsilon=0.0):
    if len(convexhullA) == 0:
        return 0
    
    A = indextovertex(g, A)
    convexhullA = indextovertex(g, convexhullA)
    
    g.vs['markwp'] = False
    pairslist = []
    for v1 in A:
        for v2 in A:
            if v1 != v2:
                pairslist.append((v1,v2))
    
    numberofwrongpoints = 0

    totalpairpaths = 0
    for pair in pairslist:
        x,y = pair
        for v in Delta(g,x,y, epsilon=epsilon):
            if v not in A and not v['markwp']:
                v['markwp'] = True
                numberofwrongpoints += 1
        totalpairpaths += len(Delta(g,x,y, epsilon=epsilon))

    g.vs['markwp'] = False
    return 1 - numberofwrongpoints / totalpairpaths


def convSanity2sampled(g, A, convexhullA, epsilon=0.0, samplepercentage=0.3):
    if len(convexhullA) == 0:
        return 0
    
    A = indextovertex(g, A)
    convexhullA = indextovertex(g, convexhullA)

    g.vs['markwp'] = False
    
    pairslist = samplepairs(A, samplepercentage)
    

    numberofwrongpoints = 0
    totalpairpaths = 0
    for pair in pairslist:
        x,y = pair
        for v in Delta(g,x,y, epsilon=epsilon):
            if v not in A and not v['markwp']:
                v['markwp'] = True
                numberofwrongpoints += 1
        totalpairpaths += len(Delta(g,x,y, epsilon=epsilon))

    g.vs['markwp'] = False
    return 1 - numberofwrongpoints / totalpairpaths

def gconfm1A(g, A, distfn=d_step, areacache=None, samplepercentage=1):
    ac = areacache
    if areacache== None:
        ac = AreaCache(g, defaultdistfn=distfn)
        #ac.precomputeDistances()
        #ac.precomputeSP()

    return ac.gconfm1A(A, samplepercentage=samplepercentage)

def gconfm1B(g, A, distfn=d_step, areacache=None, samplepercentage=1):
    ac = areacache
    if areacache== None:
        ac = AreaCache(g, defaultdistfn=distfn)
        #ac.precomputeDistances()
        #ac.precomputeSP()

    return ac.gconfm1B(A, samplepercentage=samplepercentage)

def gconfm1Aminmax(g, A, distfn=d_step, areacache=None, usemax=False, samplepercentage=1):
    ac = areacache
    if areacache== None:
        ac = AreaCache(g, defaultdistfn=distfn)
        #ac.precomputeDistances()
        #ac.precomputeSP()

    return ac.gconfm1Aminmax(A, usemax=usemax, samplepercentage=samplepercentage)



def EntropyConvexity(g, P, delta=1, distfn=d_step, areacache=None, samplepercentage=1):
    ac = areacache
    if areacache == None:
        ac = AreaCache(g, defaultdistfn=distfn)
        # ac.precomputeDistances()
        # ac.precomputeNHAreas(theta=1.0)
        # ac.precomputeNHAreas(theta=2.0)
        # ac.precomputeNHAreas(theta=3.0)
    
    return ac.EntropyConvexity(P, delta=delta, samplepercentage=samplepercentage)