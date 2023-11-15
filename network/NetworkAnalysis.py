import math
from collections import defaultdict

import pandas as pd
import networkx as nx

import utils.utils as utils
from components.Anomalies import PHENOMENA

def branching_complexity(G, X=None):
    '''
    Returns a map of nodes in G to their respective branching complexity.
    
    If a placement X is provided, branching complexities of instrumented nodes 
    will reset to 1.

    :param G: The stormwater graph
    :param X: A potential placement [(v_j,sid)] of nodes/sensors
    :return: A dictionary of nodes to their respective branching complexity
    '''
    bc = {}
    for v_j in nx.topological_sort(G):
        if G.in_degree(v_j) == 0: # Leaf node
            bc[v_j] = 1.0
        # Per phenomenon?
        elif X is not None and any(v_j == v for v,_sid in X):
            bc[v_j] = 1.0
        else: # Non-leaf node
            bnums = [bc[v_i] for v_i in G.predecessors(v_j)]
            max_bnum = max(bnums)
            bc[v_j] = max_bnum + sum(bnum / max_bnum for bnum in bnums) - 1
    return bc

def entropy(distr):
    H = 0.0
    total_area = sum(area for area in distr.values())
    for area in distr.values():
        p_x = area / total_area
        H += -p_x * math.log2(p_x)
    return H

           
def semantic_entropy_index(G, X=None, S=None, Gtrue=None):
    '''
    Returns a map of nodes in G to their respective semantic entropy index.
    
    If a placement X is provided, semantic entropy indices of instrumented nodes
    will be defined on disjoint upstream subgraphs. The set of sensors S must 
    also be provided.
    
    If G is a subgraph, then semantic land use values from Gtrue will be used. 
    
    :param G: The stormwater graph
    :param X: The potential placement [(v_j,sid)] of nodes/sensors
    :param S: The set of sensors that can be placed in the network
    :param Gtrue: A reference to the original digraph G
    :return: A dictionary of nodes to their respective semantic entropy index
    '''
    if Gtrue is None:
        Gtrue = G

    # Return the semantic entropy at every node in G
    if X is None:
        se = {}
        for v_i in G.nodes:
            # Merge land uses in all sids
            landuses = defaultdict(float)
            total_area = 0.0
            for sid in Gtrue.find_attached_subcatchments(v_i):
                for u, a in Gtrue.subc[sid]['landuse_area'].items():
                    landuses[u] += a
                    total_area += a
        
            # Compute entropy
            se[v_i] = entropy(landuses)
        return se

    # Return the semantic entropy of each instrumented node in G
    else:
        se = {}
        for p in PHENOMENA:
            X_vi = {v_i for v_i,sid in X if S[sid].phenomenon == p}
            
            visited = set()
            for v_j in nx.topological_sort(G):
                if v_j not in X_vi:
                    continue
                
                landuses = defaultdict(float)
                total_area = 0.0
                upstream_nodes = (nx.ancestors(G, v_j) | {v_j}) - visited
                for vup in upstream_nodes:
                    visited.add(vup)
                    for sid in Gtrue.find_attached_subcatchments(vup):
                        for u,a in Gtrue.subc[sid]['landuse_area'].items():
                            landuses[u] += a
                            total_area += a

                se[v_j] = entropy(landuses)
                visited.add(v_j)
        return se

def propagation_time(G, A, full=False):
    '''
    Returns a map of anomaly propagation paths to respective propagation times.
    
    Note: This function will read the propagation time cache to create this map.
    
    :param G: The stormwater graph
    :param X: The potential placement [(v_j,sid)] of nodes/sensors
    :param full: If full=True, propagation times for anomalies will not be 
                 averaged.
    :return: A dictionary of src/dst pairs to their respective propagation time
    '''
    if full:
        proptime_map = defaultdict(lambda : defaultdict(lambda : float('inf')))

        fproptime_cache = utils.get_fproptime_cache(G.fswmm, A.name)
        Xcache = pd.read_csv(fproptime_cache)
        for _,r in Xcache.iterrows():
            proptime_map[r['SrcID'],r['DstID']][r['AnomalyID']] = r['Time']

        # Proptime from a node to itself is 0
        for v in G.nodes:
            proptime_map[v,v] = defaultdict(lambda : 0)
        
    else:
        proptime_map = defaultdict(lambda : float('inf'))

        fproptime = utils.get_fproptime(G.fswmm, A.name)
        Xproptime = pd.read_csv(fproptime)
        for _,r in Xproptime.iterrows():
            proptime_map[r['SrcID'],r['DstID']] = r['Time']

        # Proptime from a node to itself is 0
        for v in G.nodes:
            proptime_map[v,v] = 0

    return proptime_map
    
def betweenness_centrality(G, A, X=None, full=True):
    '''
    Return a map of nodes in G to their respective betweenness centrality.
    
    If a placement X is provided, betweenness centralities of instrumented nodes
    will reset the count. (This effectively makes the metric count the number 
    of anomalies that would *first* be observed at the node.)
    
    :param G: The stormwater graph
    :param A: The set of anomalies that can occur in the network.
    :param X: The potential placement [(v_j,sid)] of nodes/sensors
    :param full: If full=True, betweenness centralities for anomalies will not
                 be averaged across phenomena.
    :return: A dictionary of src/dst pairs to their respective betweenness 
             centralities
    '''
    
    btn = defaultdict(int)
    
    # Find if Xproptime from G.fswmm,A.name is cached
    if hasattr(betweenness_centrality, 'Xproptime') and \
            betweenness_centrality.currG == G.fswmm and \
            betweenness_centrality.currA == A.name:
        Xproptime = betweenness_centrality.Xproptime

    else:
        betweenness_centrality.currG = G.fswmm
        betweenness_centrality.currA = A.name

        fproptime = utils.get_fproptime_cache(G.fswmm, A.name)
        Xproptime = pd.read_csv(fproptime)

        # Only consider Xproptime that are in time
        Xproptime = Xproptime[Xproptime['Time'] <= 1800] # 30 mins

        Xproptime = Xproptime.groupby(['AnomalyID','SrcID']) \
                             .nth(-1) \
                             .reset_index()

        # Save for next time
        betweenness_centrality.Xproptime = Xproptime
    
    if full: 
        # Xproptime['Phenomena'] = Xproptime['AnomalyID'].apply(lambda x : A[x].phenomena)
        instrumented = {(v,S[sid].phenomenon) for v,sid in (X or [])}
        for _,r in Xproptime.iterrows():
            src,dst = r['SrcID'], r['DstID']
            if src != 'J1': continue

            for p in A[r['AnomalyID']].phenomena:
                for path in nx.all_simple_paths(G, src, dst):
                    # Remove anomalies that would already be observed?
                    if any(v_i in instrumented for v_i in path):
                        continue

                    for v_i in path:
                        btn[v_i,p] += 1
                        # if (v_i,p) in instrumented:
                        #     break

    else: 
        instrumented = {v for v,_sid in (X or [])}
        for _,r in Xproptime.iterrows():
            src,dst = r['SrcID'], r['DstID']
            for path in nx.all_simple_paths(G, src, dst):
                
                # Remove anomalies that would already be observed?
                if any(v_i in instrumented for v_i in path):
                    continue

                for v_i in path:
                    btn[v_i] += 1
                    # if v_i in instrumented:
                    #     break
                        
    return btn

if __name__ == '__main__':
    
    from main.PARAMS import FSWMM, FSENSORS
    from components.Anomalies import Anomalies
    from components.Sensors import Sensors
    
    from network.StormwaterGraph import StormwaterGraph as SWGraph
    fswmm = FSWMM[0]
    G = SWGraph(fswmm=fswmm)
    
    S = Sensors(FSENSORS)

    fanomalies = utils.get_funiform_anomalies(fswmm)
    A = Anomalies(fanomalies, name='uniform')
    
    pt = propagation_time(G, A, full=True)
    print(pt)
    input()

    base_btn = betweenness_centrality(G, A, full=False)
    X = [('J5',1)]
    btn = betweenness_centrality(G, A, X=X, full=False)

    print('BTN (base):', sorted(base_btn.items()))
    print('BTN       :', sorted(btn.items()))
    print()
    
    base_btn = betweenness_centrality(G, A, full=True)
    X = [('J5',1)]
    btn = betweenness_centrality(G, A, X=X, full=True)

    print('BTN (base):', sorted(base_btn.items()))
    print('BTN       :', sorted(btn.items()))
    
    
