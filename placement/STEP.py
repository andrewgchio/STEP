from os.path import exists
from collections import defaultdict
from random import randrange
from itertools import product
from collections import Counter

import pandas as pd
import networkx as nx
from tqdm import tqdm

import metis
from pulp import *
import cyipopt
from cyipopt import minimize_ipopt 

import utils.utils as utils
from main.PARAMS import *
from components.Sensors import Sensors
from components.Anomalies import Anomalies, getAnomaliesAt

from network.StormwaterGraph import StormwaterGraph as SWGraph, upstream_subgraphs, upstream_subgraphs_count
from network.NetworkAnalysis import betweenness_centrality, propagation_time, branching_complexity, semantic_entropy_index
from network.Objectives import *

class STEP:
    
    def __init__(self, G, S, A):
        self.G, self.S, self.A = G, S, A
    
    def do_placement(self, n_partitions, n_trace, cache=None):
        X = []
        
        # using btn
        # btn_cache = utils.get_fstep_betweenness_centrality_partitions(self.G.fswmm, A.name)
        # if not exists(btn_cache):
        #     btn_cache = None
        #
        # # using bc
        # bc_cache = utils.get_fstep_branching_complexity_partitions(self.G.fswmm)
        # if not exists(bc_cache):
        #     bc_cache = None

        # Use one of the caches: btn_cache or bc_cache
        for v_part, Gsub in self.partition_graph(n_partitions):
            
            # In the partitioned graph, we want to limit the branching complexity
            print(v_part, str(Gsub))
            
            # bc = branching_complexity(Gsub, X)
            # total_A_v = 0
            # for v in Gsub.nodes:
            #     A_v = getAnomaliesAt(A.name, v)
            #     total_A_v += len(A_v)
            #
            # if v_part is not None:
            #     print('  Upstream: ', len(nx.ancestors(G, v_part)), f'; |A(v)| = {total_A_v}')
            # else:
            #     print('  v_part is None', f'; |A(v)| = {total_A_v}')
            #
            # for v in Gsub.nodes:
            #     A_v = getAnomaliesAt(A.name, v)
            #     print(f'  {v} : |A(v)|={len(A_v)}')
            
            # Determine amount of budget to allocate for Gsub
            # Bc = self._compute_budget(Gsub, B)
            
            # Find the optimal placement for each subgraph 
            Xsub = self.find_optimal_placement(Gsub, v_part, n_trace//n_partitions)
            
            X.extend(Xsub)
        
        # X = self.adjust_placements(X)
        
        # Save results into cache
        if cache:
            with open(cache, 'w') as f:
                f.write('NodeID,SensorID,npart,ntrace\n')
                for v,sid in X:
                    f.write(f'{v},{sid},{n_partitions},{n_trace}\n')
        
        X = self.adjust_placements(X)

        if cache:
            cache2 = utils.add_to_fname(cache, append='_adjusted')
            with open(cache2, 'w') as f:
                f.write('NodeID,SensorID,npart,ntrace\n')
                for v,sid in X:
                    f.write(f'{v},{sid},{n_partitions},{n_trace}\n')

        return X
    
    def partition_graph_metis(self, n_partitions):
        # Partition the graph using metis
        _objval, partitions = metis.part_graph(self.G, n_partitions)
        
        # Create subgraphs based on the partitioning
        subgraphs = []
        for i in range(n_partitions):
            Vsub = [v for v,label in zip(self.G.nodes, partitions) if label == i]
            subgraphs.append(self.G.subgraph(Vsub).copy())
        
        return subgraphs
    
    def partition_graph(self, n_partitions=8, cache=None):
        '''
        Partition the graph using the betweenness centrality metric.
        
        Reads the partitioning from a cached file
        '''
        assert n_partitions >= 2, 'Must have at least 2 partitions'

        # Return value from the cache
        # if cache is not None:
        #     Xnodes = pd.read_csv(cache, nrows=n_partitions-1)
        #     print(Xnodes)
        #     X_v = [r['NodeID'] for _,r in Xnodes.iterrows()]
        #     for Gsub in upstream_subgraphs(self.G, X_v):
        #         vpart = next((v for v in Gsub.nodes if v in X_v), None)
        #         yield vpart, Gsub
        #     return
        
        # Try just selecting the node that has the highest betweenness???
        X = []
        for i in range(n_partitions):
            
            btn = betweenness_centrality(self.G, self.A, X, full=False)
            print(sorted(Counter(btn).items(),key=lambda x : -x[1])[:15])
            
            best_score, best_node = (float('-inf'), float('inf')), None
            
            for v in tqdm(nx.topological_sort(self.G), total=self.G.number_of_nodes()):
                
                # Already chosen
                if any(v == x for x,_ in X):
                    continue
                
                # Try the placement
                X.append((v,-1))
                
                # Evaluate the placement 
                # new_btn = betweenness_centrality(G, A, X, full=False)
                # proposed_btn = max(new_btn.values())
                
                # If the score is the same for two nodes, then use the one that
                # would split the graph into more even sized subgraphs
                score = btn[v], self.compute_evenness(X)
                
                if best_score < score:
                    best_score, best_node = score, v
                
                # Reset the placement
                X.pop()
            
            # Add best node
            # if best_node is None:
            #     break # Nothing found
            
            X.append((best_node,-1))
            
            # print(f'Best reduction ({i}): {best_score} at node {best_node} (prev total={-best_score[0]+base_btn})')
            print(f'Best reduction ({i}): {best_score} at node {best_node}')
            # print(sorted(Counter(btn).items(), key=lambda x : -x[1]))
            print('Current subgraph partitions: ')
            X_v = [v for v,_ in X]
            subgraphs_count = upstream_subgraphs_count(self.G, X_v)
            print(subgraphs_count)
            print()

    
        X_v = [v for v,_ in X]
        for Gsub in upstream_subgraphs(self.G, X_v):
            vpart = next((v for v in Gsub.nodes if v in X_v), None)
            yield vpart, Gsub
        return
        

    def compute_evenness(self, X):
        X_v = list(map(lambda x : x[0], X))
        Vsub_sizes = np.array(upstream_subgraphs_count(self.G, X_v))
        return -np.sum(np.abs(Vsub_sizes - self.G.number_of_nodes()/(1+len(X_v))))

    def find_optimal_placement_pulp(self, Gsub, B):
        model = LpProblem('SensorPlacement', LpMinimize)
        
        Xflat = list(enumerate(product(self.S.get_sids(), nx.topological_sort(Gsub))))
        X = LpVariable.dicts('X', (self.S.get_sids(), nx.topological_sort(self.G)), 
                             0,1, LpBinary)
        
        # Objective
        # model += lpSum( (X[sid][v]*0 + (1-X[sid][v]) * bc[v]) for sid in Sids for v in V) # v1
        
        # Constraints
        # model += lpSum(X[sid][v] * S[sid] for sid,_ in S for v in Gsub.nodes) <= B

    def find_optimal_placement_ipopt(self, Gsub, B, cache=None):
        Xflat = list(enumerate(product(self.S.get_sids(), nx.topological_sort(Gsub))))
        
        # Objective - optimize on betweenness centrality
        btn = betweenness_centrality(self.G, self.A)
        
        x_o = {(sid,aid) : s_l.phenomenon in a_k.phenomena 
               for sid,s_l in self.S for aid,a_k in self.A}
        pt = propagation_time(self.G, self.A, full=False)
        x_t = {(v_i,v_j) : pt[v_i,v_j] <= 1800 
               for v_i in self.G.nodes
               for v_j in self.G.nodes}
    
        def objective(X):
            covered_set = set()
            
            
            return np.sum([(1-X[i]) * x_t[v_i,v_j]
                           for v_i in self.G.nodes
                           for i,(sid,v_j) in Xflat])
                # for i,(sid,v_j) in Xflat:
                    # total += X[i] # * x_o[sid,aid] # * x_t[aid,v_j]
            # return total
            # return np.sum([(1-X[i])*btn[self.S[sid].phenomenon,v] for i,(sid,v) in Xflat])

        # Constraints - Budget
        def constraint(X):
            return B - np.sum([X[i] * self.S[sid].cost for i,(sid,_v) in Xflat]) 
    
        # Define the bounds on the optimization variables
        lb = [0] * len(Xflat)
        ub = [1] * len(Xflat)

        # Define the initial guess
        x0 = [1] * len(Xflat)

        # Define the problem bounds
        problem_bounds = [(lb[i], ub[i]) for i in range(len(Xflat))]

        # Define the problem constraints
        problem_constraints = [{'type': 'ineq', 'fun': constraint}]

        # Minimize the objective function using IPOPT
        solution = minimize_ipopt(
            objective,
            x0,
            bounds=problem_bounds,
            constraints=problem_constraints,
            options={'print_level': 5,
                     'tol' : 1e-4}
        )
        
        if cache:
            
            flog = utils.add_to_fname(cache, append='_log')
            
            with open(flog, 'a') as f:
                f.write('NodeID,SensorID,VarValue,RoundedVar,Budget\n')
                for val,(i,(sid,v)) in zip(map(float, solution['info']['x']), Xflat):
                    f.write(f'{v},{sid},{val},{round(val)},{B}\n')
                f.write('>>>>>\n')
                f.write(str(solution))
                f.write('\n<<<<<\n')

        # Convert into format
        X = []
        for val,(i,(sid,v)) in zip(map(float, solution['info']['x']), Xflat):
            if round(val) == 1:
                X.append((v,sid))
        return X
    
    def find_optimal_placement(self, Gsub, v_part, n_trace):
        Xsub = []
        
        # Put ALL sensors at the partition node, if exists
        if v_part is not None:
            for sid,_s_l in self.S:
                Xsub.append((v_part,sid))

        # total_cost = sum(self.S[sid].cost for _v,sid in Xsub)
        
        base_bc = np.mean(list(branching_complexity(Gsub, Xsub).values()))

        # Find n_trace number of new placements
        for _ in range(n_trace):
        
            # Find location that would most decrease branching complexity
            best_score, best_v = float('-inf'), None
            for v_j in Gsub.nodes:
                
                # Skip the already instrumented partition node
                if v_j == v_part:
                    continue
                
                # Try placement
                Xsub.append((v_j, -1))
                
                bc = np.mean(list(branching_complexity(Gsub, Xsub).values()))
                score = base_bc - bc
                
                if best_score < score:
                    best_score, best_v = score, v_j
                
                # Reset placement
                Xsub.pop()
            
            # Find sensor that would 
            best_score, best_sid = float('-inf'), None
            for sid,_s_l in self.S:
                
                # Try placement
                Xsub.append((best_v, sid))
                
                score = semantic_entropy_index(Gsub, S=self.S, Gtrue=self.G)[best_v]
                
                if best_score < score:
                    best_score, best_sid = score, sid
                
                # Reset placement
                Xsub.pop()
            
            # Place the sensor, if one was properly found
            if best_v is not None and best_sid is not None:
                Xsub.append((best_v, best_sid))
        
        return Xsub
            
        
        # # Greedily choose more sensors to add in
        # while True:
        #
        #     base_bc = np.mean(list(branching_complexity(G, Xsub).values()))
        #     best_score, best_x = float('-inf'), None
        #
        #     # Find best v_j to put a sensor s_l
        #     for v_j in Gsub.nodes:
        #         for sid,_s_l in self.S:
        #
        #             # Ignore if already in placement
        #             if (v_j,sid) in Xsub:
        #                 continue
        #
        #             # Try the placement
        #             Xsub.append((v_j,sid))
        #
        #             bc = branching_complexity(G, Xsub)
        #             score = base_bc - np.mean(list(bc.values()))
        #
        #             if best_score < score:
        #                 best_score, best_x = score, (v_j,sid)
        #
        #             # Reset the placement
        #             Xsub.pop()
        #
        #     # Nothing was found
        #     if best_x is None:
        #         break
        #
        #     Xsub.append(best_x)
        #     total_cost += S[best_x[1]].cost
        #
        #     # Round up the number of sensors that can be placed
        #     if total_cost > B:
        #         break
        #
        # return Xsub
    
    def _compute_budget(self, Gsub, B):
        '''
        Return the rough budget that should be allocated for placing a sensor in
        the subgraph. 
        '''
        # V2 - based on the expected number of anomalies
        return sum(len(getAnomaliesAt(self.A.name, v)) for v in Gsub.nodes) / len(self.A) * B
        
        # V1 - based on the number of nodes
        # return Gsub.number_of_nodes() / self.G.number_of_nodes() * B
        
    def adjust_placements(self, X):
        '''
        Adjust the placement of each item in X, in case a neighboring node would
        provide a better overall placement
        '''
        Xadj = [(v,sid) for (v,sid) in X]
        btn = betweenness_centrality(self.G, self.A, full=True)
        for i,(v,sid) in enumerate(X):
            best_value, best_xadj = btn[self.S[sid].phenomenon, v], None
            
            for n in self.G.neighbors(v):
                
                # Check if n is already measured, and skip if so
                if (n,sid) in X:
                    continue
                
                # Make adjust
                Xadj[i] = (n,sid)
                
                # Do evaluation - maximize betweenness centrality
                value = btn[self.S[sid].phenomenon, v]
                
                if value > best_value:
                    best_value, best_xadj = value, (n,sid)
            
            # switch it to the best one found
            if best_xadj is not None:
                Xadj[i] = best_xadj
            else: # Revert back
                Xadj[i] = X[i]
        
        return Xadj

def get_STEP_placement_budget(fstep, S, B):
    Xcache = pd.read_csv(fstep)
    Xcache['X'] = Xcache.apply(lambda x : (x['NodeID'], x['SensorID']), axis=1)
    Xcache = Xcache.groupby('n').agg({'X' : lambda x : list(x)}).reset_index()

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
    
    X = []
    for v,sid in Xcache['X'].iloc[0]:
        X.append((v,sid))
            
    return X

def get_STEP_placement_instr(fstep, S, n):
    Xcache = pd.read_csv(fstep)
    
    # Take n that you want
    # Careful: there can be an error when n is not found
    
    Xcache = Xcache[Xcache['n'] == n]
    
    if Xcache.shape[0] == 0:
        raise f'Error: n({n}) is not found'
    
    X = []
    for _,r in Xcache.iterrows():
        X.append((r['NodeID'], r['SensorID']))
    return X

# if __name__ == '__main__':
#
#     return
#
#     fswmm = FSWMM[5]
#     # for fswmm in FSWMM:
#     G = SWGraph(fswmm=fswmm)
#
#     S = Sensors(FSENSORS)
#
#     fanomalies = utils.get_fsemantic_anomalies(fswmm)
#     A = Anomalies(fanomalies, name='semantic')
#
#     # funiform_anomalies = utils.get_funiform_anomalies(fswmm)
#     # A = Anomalies(funiform_anomalies, name='uniform')
#
#     N = range(10,160, 10) # Number of locations to instrument
#
#     for n in N:
#
#         step = STEP(G, S, A)
#
#         cache = utils.get_fstep(fswmm, A.name, n)
#         X = step.do_placement(n_partitions=n//2, n_trace=n//2, cache=cache)
#
#         pt = propagation_time(G, A, full=True)
#
#         print('Budget used', sum(S[sid].cost for _,sid in X))
#         # print('Coverage from placement: ', coverage(X, pt, G, S, A))
#         print('Anomaly Coverage from placement', (1-count_missed(X, pt, S, A)) * 100)
#         print('Traceability from placement: ', physical_traceability(X, pt, G, S, A)/G.number_of_nodes() * 100)
#         print()
#         print()
#
#         # input('Enter to go to next placement...')
#
#         # break
