import numpy as np
import networkx as nx

from network.NetworkAnalysis import propagation_time
from components.Anomalies import getAnomaliesAt
from components.Sensors import getSensorsMeasuring
from network.StormwaterGraph import upstream_subgraphs, upstream_subgraphs_count 

def coverage(X, pt, G, S, A, tol=0.5):
    '''
    Compute the average proportion of the network that is effectively covered by
    the placement X
    '''

    def covered(v_i):
        '''
        p% of anomalies originating at v_i can be detected

        Returns 1 if the node is covered, and 0 otherwise
        '''
        detected = 0
        anomalies_at_v_i = getAnomaliesAt(A.name, v_i)

        for aid,a_k in anomalies_at_v_i.items():
            detected += any(
                ((v_i,v_j) in pt or v_i == v_j) and \
                S[sid].phenomenon in a_k.phenomena and \
                pt[v_i,v_j][aid] <= 3600 # 60 mins * 60 sec/min
                for v_j,sid in X)

        return 0 if detected <= tol * len(anomalies_at_v_i) else 1

    return sum(covered(v_i) for v_i in G.nodes) / G.number_of_nodes()

def priority(X, pt, G):
    '''
    Compute the average priority of the network that is effectively covered by
    the placement X
    '''
    def priority(v_j):
        '''
        Total priority of nodes that are observed 
        '''
        total = 0
        for v_i in nx.ancestors(G, v_j):
            if pt[v_i,v_j] <= 3600:
                for sid in G.find_attached_subcatchments(v_i):
                    total += G.subc[sid]['landuse_weight']
        return total
    
    return sum(priority(v_j) for v_j,_sid in X)
    

def physical_traceability(X, pt, G, S, A):
    '''
    Compute the average number of nodes in the network that can be effectively 
    eliminated when an anomaly is detected. If the anomaly is not detected, then
    we assume the number of nodes in the graph G.
    '''
    def number_of_nodes_eliminated(a_k):
        # Find first node to detect a_k - find the node in X which has the 
        # minimum propagation time
        ptime,v_j = min(((pt[a_k.origin,v_j][a_k.aid], v_j) for v_j,sid in X 
                   if S[sid].phenomenon in a_k.phenomena and \
                   pt[a_k.origin,v_j][a_k.aid] <= 3600), 
                key=lambda x : x[0], # Do not use v_j as a tie-breaker
                default=(float('inf'),None))
        if ptime == float('inf'):
            return 0
        else:
            return G.number_of_nodes() - len(nx.ancestors(G, v_j)) - 1 # {v_j}
    
    return sum(number_of_nodes_eliminated(a_k) for _,a_k in A) / len(A)

def count_missed(X, pt, S, A):
    '''
    Compute the proportion of anomalies that are detected by the placement X
    '''
    detected = 0
    for aid,a_k in A:
        detected += any(
            ((a_k.origin,v_j) in pt or a_k.origin == v_j) and \
            S[sid].phenomenon in a_k.phenomena and \
            pt[a_k.origin,v_j][aid] <= 1800 # 30 mins * 60 min/sec
            for v_j,sid in X)
    return 1 - detected / len(A)

# def physical_traceability(X, G, S, shortcut=False):
#     '''
#     Percentage of locations that can be eliminated 
#     Number of nodes upstream of each placed sensor
#
#     shortcut: when optimizing traceability greedily, duplicate nodes will be 
#     selected, since Anomalies are not considered during traceability. Therefore,
#     we can take a shortcut and just assume all types of sensors are added in
#     at each location
#     '''
#     tr = []
#     for p in sorted({s_l.phenomenon for _,s_l in S}):
#         S_p = getSensorsMeasuring(p) 
#         X_v = {v for v,s in X if s in S_p}
#
#         Gsubs_sizes = upstream_subgraphs_count(G, X_v)
#
#         # print(f'For p={p}')
#         # print(X_v, S_p)
#         # print(f'len(Gsubs)={Gsubs_sizes}')
#
#         # We have a little weight from this 
#         tr_s = max(Gsubs_sizes) + 0.001 * np.mean(Gsubs_sizes)
#
#         tr.append(tr_s)
#
#         if shortcut:
#             break
#
#     # trying to minimize the number of nodes
#     return -np.mean(tr)

def semantic_traceability(X, G, S, thresh=1, shortcut=False):
    '''
    Number of semantic land uses upstream of each placed sensor
    
    Note: this will only start playing a part after the branching complexity
    has started partitioning the graphs a lot more
    '''
    tr = []
    for p in sorted({s_l.phenomenon for _,s_l in S}):
        S_p = getSensorsMeasuring(p)
        X_v = {v for v,s in X if s in S_p}
        
        Gsubs = upstream_subgraphs(G, X_v)
        
        slu = set()
        # Count semantic land uses in each subgraph
        for Gsub in Gsubs:
            for node in Gsub.nodes:
                sids = G.find_attached_subcatchments(node)
                for sid in sids:
                    landuses = G.subc[sid]['landuse_area']
                    for landuse,area in landuses.items():
                        if area > thresh:
                            slu.add(landuse)
        
        tr.append(len(slu))
        
        if shortcut:
            break
    
    return -np.mean(tr)


# def traceability(X, G, S):
#     '''
#     Average between the physical and semantic traceability objectives
#     '''
#     return physical_traceability(X, G, S)
#
#     # return 0.5 * physical_traceability(X, G, S, shortcut=shortcut) + \
#     #         0.5 * semantic_traceability(X, G, S, shortcut=shortcut)

def avg_detection_time(X, A, G, S, penalty=3600):
    
    pt = propagation_time(G, A, full=True)
    detection_time = 0
    for aid,a_k in A:
        detection_time += min(
            pt[a_k.origin,v_j][aid]
                if ((a_k.origin,v_j) in pt or a_k.origin == v_j) and \
                aid in pt[a_k.origin,v_j] and \
                S[sid].phenomenon in a_k.phenomena else \
            penalty
            for (v_j,sid) in X)

    # negative so that we can just try to maximize the objective
    return -detection_time / len(A)
