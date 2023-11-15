import random 
from os.path import exists
from copy import deepcopy

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

import utils.utils as utils

from components.Anomalies import Anomalies
from components.Sensors import Sensors

from network.Objectives import coverage, priority, physical_traceability, semantic_traceability
from network.NetworkAnalysis import propagation_time

class GeneticAlgorithm:
    
    def __init__(self, G, S, A):
        self.G = G
        self.S = S
        self.A = A
        
        # Cached values
        self.V = list(nx.topological_sort(G))
    
    def do_placement(self, B, fitness, n_pop=500, n_iter=200, r_cross=0.8, r_mut=0.01, cache=None):
        pop = self.initial_population(B, n_pop)
        
        best_value, best_placement = float('-inf'), None

        change_detected = 0 # if not change detected in 3 rounds, then stop
        
        if cache is not None:
            f = open(cache, 'w')
            f.write('NodeID,SensorID,Gen,Budget\n')
        
        for gen in range(n_iter):
            
            print(f'Starting generation {gen}')
            
            print('Updating best scores')
            scores = []
            for x in tqdm(pop): # enable tqdm
                scores.append(fitness(x))
            
            for x,score in zip(pop, scores):
                if score > best_value:
                    best_placement, best_value = x,score
                    print(f'  new best = {best_value}', flush=True)
                    change_detected = 0

            # Create the next generation
            print('Selecting parents for next generation')
            selected = [self.selection(pop, scores) for _ in range(n_pop)]

            children = []
            for i in tqdm(range(0, n_pop, 2)):
                # parent pairs
                X1, X2 = selected[i], selected[i+1]
                
                # do crossover and mutation
                for X in self.crossover(X1,X2, B, r_cross):
                    self.mutation(X, B, r_mut)
                    children.append(X)

            pop = children
            change_detected += 1

            if cache is not None:
                for v,s in X:
                    f.write(f'{v},{s},{gen},{B}\n')
                f.flush()
            
            # If there is no change for a few generations, then stop early
            if change_detected == 5:
                break
    
        print('Best value for algorithm: ', best_value)
        

        return best_placement, best_value
        
    def initial_population(self, B, n_pop):
        return [self.get_random_placement(B) for _ in range(n_pop)]

    def get_random_placement(self, B):
        X = []
        total_cost = 0
        while True:
            x = (self.get_random_node(), self.get_random_sensor())

            # The next placement would be too expensive
            if total_cost + self.S[x[1]].cost > B:
                return X

            # The next placement is already there
            if x not in X:
                X.append(x)
                total_cost += self.S[x[1]].cost

    def get_random_node(self):
        return random.choice(self.V)

    def get_random_sensor(self):
        return random.choice(self.S.get_sids())

    def selection(self, pop, scores, k=3):
        '''
        Tournament Selection: Choose k individuals from the population. Then,
        return the one with the maximum score.
        '''
        ix = np.random.choice(range(len(pop)), size=k, replace=False)
        return max((scores[i], pop[i]) for i in ix)[1]

    def crossover(self, x1, x2, B, r_cross=0.8):
        '''
        Perform crossover on two individuals of the population
        '''
        c1, c2 = x1.copy(), x2.copy()
        if random.random() < r_cross:
            minlen = min(len(x1), len(x2))
            pt = random.randint(1,minlen-2)

            c1 = x1[:pt] + x2[pt:]
            c2 = x2[:pt] + x1[pt:]
            
            # Ensure that placements are still valid
            # c1 = remove_random_sensors(c1, self.S, B)
            # c2 = remove_random_sensors(c2, self.S, B)

        return c1, c2

    def compute_total_sensor_cost(self, X):
        return sum(self.S[sid].cost for _,sid in X)

    def remove_random_sensors(self, X, B):
        while self.compute_total_sensor_cost(X) > B:
            X.pop(np.random.choice(len(X)))
        return X

    def mutation(self, X, B, r_mut=0.01):
        for i in range(len(X)):
            if random.random() < r_mut:
                # Just replace something
                X[i] = (self.get_random_node(), self.get_random_sensor())

def genetic_coverage(G, S, A, B, cache=None):
    ga = GeneticAlgorithm(G, S, A)
    pt = propagation_time(G, A, full=True)
    fitness = lambda x: coverage(x, pt, G, S, A)
    ga.do_placement(B, fitness, cache=cache)

def genetic_priority(G, S, A, B, cache=None):
    ga = GeneticAlgorithm(G, S, A)
    pt = propagation_time(G, A, full=False)
    fitness = lambda x: priority(x, pt, G)
    ga.do_placement(B, fitness, cache=cache)

# Not used
# def genetic_traceability(G, S, A, B, cache=None):
#     ga = GeneticAlgorithm(G, S, A)
#     pt = propagation_time(G, A, full=True)
#     fitness = lambda x : traceability(x, pt, G, S, A)
#     ga.do_placement(B, fitness, cache=cache)

def genetic_ph_traceability(G, S, A, B, cache=None):
    ga = GeneticAlgorithm(G, S, A)
    pt = propagation_time(G, A, full=True)
    fitness = lambda x : physical_traceability(x, pt, G, S, A)
    ga.do_placement(B, fitness, cache=cache)

def genetic_se_traceability(G, S, A, B, cache=None):
    ga = GeneticAlgorithm(G, S, A)
    pt = propagation_time(G, A, full=True)
    fitness = lambda x : semantic_traceability(x, pt, G, S, A)
    ga.do_placement(B, fitness, cache=cache)

def get_genetic_placement_budget(fgenetic, S, B):
    Xcache = pd.read_csv(fgenetic)
    Xcache['X'] = Xcache.apply(lambda x : (x['NodeID'], x['SensorID']), axis=1)
    Xcache = Xcache.groupby('Budget').agg({'X' : lambda x : list(x)}).reset_index()

    Xcache['B'] = Xcache['X'].apply(lambda x : len(set(v for v,_ in x)))
    
    def compute_cost(X):
        total_cost = 0
        locations_instrumented = set()
        for v_j,sid in X:
            total_cost += S[sid].cost
            locations_instrumented.add(v_j)
        total_cost += len(locations_instrumented) * 400
        return total_cost
    
    Xcache['Cost'] = Xcache['X'].apply(compute_cost)
    
    # Get closest number of nodes B
    Bs = list(Xcache['Cost'])
    B = Bs[min((abs(Bcache-B),i) for i,Bcache in enumerate(Bs))[1]]

    Xcache = Xcache[Xcache['Cost'] == B]

    # Instead, return all sensors at specified nodes
    X = []
    for v,sid in Xcache['X'].iloc[0]:
        X.append((v,sid))
            
    return X

def get_genetic_placement_instr(fgenetic, S, n):
    Xcache = pd.read_csv(fgenetic)
    Xcache['X'] = Xcache.apply(lambda x : (x['NodeID'], x['SensorID']), axis=1)
    
    Xcache = Xcache.groupby('Budget').agg({'X' : lambda x : list(x)}).reset_index()
    
    Xcache['n'] = Xcache['X'].apply(lambda x : len(set(v for v,_ in x)))

    # Get closest number of nodes n
    Ns = list(Xcache['n'])
    n = Ns[min((abs(Bcache-n),i) for i,Bcache in enumerate(Ns))[1]]
    
    Xcache = Xcache[Xcache['n'] == n]
    
    # return Xcache['X'].iloc[0]
    
    # Instead, return all sensors at specified nodes
    X = []
    for v,_sid in Xcache['X'].iloc[0]:
        for sid,_ in S:
            X.append((v,sid))
            
    return X
            
            