import random
from os.path import exists
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from components.Anomalies import Anomalies, getAnomaliesAt

from network.NetworkAnalysis import propagation_time
from network.NetworkAnalysis import branching_complexity

from network.Objectives import coverage, priority
from network.Objectives import physical_traceability, semantic_traceability

def greedy_placement(G, S, A, B, evaluate_placement, cache=None):
    '''
    Template for the Greedy Heuristic Placement Algorithm
    
    Note: evaluate_placement should produce a maximum value
    '''
    
    curr_placement = [] # [(v_j,s_l)]
    curr_placement_hash = set() # {(v_j,sid)}
    total_cost = 0 # for the sensors so far

    if cache is not None:
        f = open(cache, 'w')
        f.write('v_j,sid,val,cost\n')
        
    # If we just add random half of the nodes, all sensors, then we hsould
    # have around 50% coverage
    # from random import sample
    # for v in sample(list(G.nodes), 200):
    #     for i in [1,2,3,4,5]:
    #         curr_placement.append((v, i))
    
    # Continue trying to add sensors
    while True:

        best_value, x = float('-inf'), None

        # Find best v_j to put a sensor
        for v_j in tqdm(G.nodes):
            for sid,s in S:

                # Sensor is too expensive, and so cannot be chosen
                if total_cost + s.cost > B:
                    continue

                # (v_j,s_l) is already in placement
                if (v_j,sid) in curr_placement_hash:
                    continue

                # Try the placement
                curr_placement.append((v_j,sid))
                new_value = evaluate_placement(curr_placement)
                if new_value > best_value:
                    best_value, x = new_value, (v_j,sid)
                    print(f'{x} => {best_value}')

                # Remove the placement for the next iteration
                curr_placement.pop()
        
        # All additional sensors would be too expensive or already used
        if x is None:
            break

        curr_placement.append(x)
        curr_placement_hash.add(x)
        total_cost += S[x[1]].cost
        if cache is not None:
            f.write(f'{x[0]},{x[1]},{best_value},{total_cost}\n')
            f.flush()
        
        # i.e., if a new sample should be taken
        if hasattr(evaluate_placement, 'reset'):
            evaluate_placement.reset()

        # print('\n'.join(list(map(lambda q : str(q[1]), A))))
        print(f'currently considering: {curr_placement} for cost {best_value}')

    if cache is not None:
        f.close()

    return best_value, curr_placement
    
def greedy_coverage(G, S, A, B, cache=None):
    pt = propagation_time(G, A, full=True)
    heuristic = lambda x: coverage(x, pt, G, S, A)
    return greedy_placement(G, S, A, B, heuristic, cache=cache)

def greedy_radius_coverage(G, S, A, B, cache=None):
    radius_coverage = {v : G.find_nodes_within_radius(v,5000) for v in G.nodes}
    def heuristic(X):
        covered = set()
        for v,_ in X:
            covered |= radius_coverage[v]
        return len(covered)
    return greedy_placement(G, S, A, B, heuristic, cache=cache)

def greedy_centrality_coverage(G, S, A, B, cache=None):
    centrality_coverage = nx.betweenness_centrality(G)
    def heuristic(X):
        return sum(centrality_coverage[v] for v,_sid in X)
    return greedy_placement(G, S, A, B, heuristic, cache=cache)

def greedy_priority(G, S, A, B, cache=None):
    pt = propagation_time(G, A, full=False)
    heuristic = lambda x: priority(x, pt, G)
    return greedy_placement(G, S, A, B, heuristic, cache=cache)

def greedy_ph_traceability(G, S, A, B, cache=None):
    # print(G.fswmm, '=>', int(len(A)/2))
    pt = propagation_time(G, A, full=True)
    
    n_sample = min(200, int(len(A)/2))
    Alist = list(map(lambda x : x[0], A))
    Aidx = random.sample(Alist, n_sample)
    Asample = [(aid,A[aid]) for aid in list(Aidx)]
    
    def heuristic(x):
        return physical_traceability(x, pt, G, S, Asample)
    
    def heuristic_reset():
        nonlocal Asample
        Aidx = random.sample(Alist, n_sample)
        Asample = [(aid,A[aid]) for aid in list(Aidx)]
    heuristic.reset = heuristic_reset
        
    return greedy_placement(G, S, A, B, heuristic, cache=cache)


# Not used
# def greedy_se_traceability(G, S, A, B, cache=None):
#     print('greedy_se_traceability is not used')
#     return
#     pt = propagation_time(G, A, full=True)
#     heuristic = lambda x : semantic_traceability(x, pt, G, S, A)
#     return greedy_placement(G, S, A, B, heuristic, cache=cache)

# Not used
# def greedy_traceability(G, S, A, B, cache=None):
#     print('greedy_traceability is not used')
#     return
#     pt = propagation_time(G, A, full=True)
#     heuristic = lambda x : traceability(x, pt, G, S, A)
#     return greedy_placement(G, S, A, B, heuristic, cache=cache)

# def greedy_detection_time(G, S, A, B, cache=None):
#     pt = propagation_time(G, A, full=True)
#     return greedy_placement(G, S, A, B, avg_detection_time, A, pt, cache=cache)


def get_greedy_placement_budget(fgreedy, S, B):
    Xcache = pd.read_csv(fgreedy)
    
    total_cost = 0
    locations_instrumented = set()
    
    X = []
    for _,r in Xcache.iterrows():
        
        added_cost = S[r['sid']].cost
        if r['v_j'] not in locations_instrumented:
            added_cost += 400 # Set cost per year
        
        if total_cost + added_cost > B:
            break
        
        X.append((r['v_j'], r['sid']))
        locations_instrumented.add(r['v_j'])
        total_cost += added_cost

        # Add all sensors at node
        # for sid,_ in S:
        #    X.append((r['v_j'], sid))
    
    # print(f'Greedy Placement Budget: For budget {B}, X = {X}')

    return X


def get_greedy_placement_instr(fgreedy, S, n):
    Xcache = pd.read_csv(fgreedy)
    # Xcache['X'] = Xcache.apply(lambda x : (x['v_j'], x['sid']), axis=1)
    X = []
    for _,r in Xcache.iterrows():
        if len(set(v for v,_ in X)) > n: 
            break

        # X.append((r['v_j'], r['sid']))

        # Add all sensors at node
        for sid,_ in S:
            X.append((r['v_j'], sid))
    return X
        
