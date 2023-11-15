from random import choice
from network.NetworkAnalysis import branching_complexity, betweenness_centrality

class GreedyBranchingComplexityPlacement:
    
    @staticmethod
    def do_placement(G, sensors, budget):
        '''
        Return a map of locations to deployed sensors
        
        :param G: the stormwater graph
        :param sensors: the set of sensor objects to place
        :param budget: the budget
        '''
        
        # Compute branching complexity difference
        def max_bc_diff(bc, placement):
            diff = bc.copy()
            for n,v in bc.items():
                pred = list(G.predecessors(n))
                if pred:
                    bc_predmax = max(bc[v] for v in G.predecessors(n))
                    diff[n] = bc[n] - bc_predmax
                else:
                    diff[n] = 0
            
            return max(((k,v) for k,v in diff.items() if k not in placement), 
                       key=lambda x : x[1])[0]

        placement = {}
        sensor_total_cost = sum(s.cost for s in sensors)
        cost_so_far = 0

        while cost_so_far + sensor_total_cost <= budget:
            bc = branching_complexity(G, placement)
            node = max_bc_diff(bc, placement)

            if node in placement: # Try another node 
                continue

            placement[node] = 'WaterQuality-A' # put a sensor here
            cost_so_far += sensor_total_cost
        
        return placement
    
    @staticmethod
    def is_deterministic():
        '''
        Return true if the algorithm is deterministic
        '''
        return True

class GreedyBetweennessCentralityPlacement:
    
    @staticmethod
    def do_placement(G, sensors, budget):
        '''
        Return a map of locations to deployed sensors
        
        :param G: the stormwater graph
        :param sensors: the set of sensor objects to place
        :param budget: the budget
        '''
        
        # Compute coverage difference
        def max_bc_diff(bc, placement):
            diff = bc.copy()
            for n,v in bc.items():
                pred = list(G.predecessors(n))
                if pred:
                    bc_predmax = max(bc[v] for v in G.predecessors(n))
                    diff[n] = bc[n] - bc_predmax
                else:
                    diff[n] = 0
            
            return max(((k,v) for k,v in diff.items() if k not in placement), 
                       key=lambda x : x[1])[0]

        placement = {}
        sensor_total_cost = sum(s.cost for s in sensors)
        cost_so_far = 0

        while cost_so_far + sensor_total_cost <= budget:
            bc = betweenness_centrality(G, placement)
            node = max_bc_diff(bc, placement)

            if node in placement: # Try another node 
                continue

            placement[node] = 'WaterQuality-A' # put a sensor here
            cost_so_far += sensor_total_cost
        
        return placement
    
    @staticmethod
    def is_deterministic():
        '''
        Return true if the algorithm is deterministic
        '''
        return True
