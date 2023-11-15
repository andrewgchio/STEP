
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from hymo import hymo

import utils.utils as utils
from main.PARAMS import *
from network.StormwaterGraph import StormwaterGraph as SWGraph
from components.Anomalies import Anomalies, Anomaly, PHENOMENA

class AnomalyGeneration2:

    def __init__(self, fswmm):
        self.graph = SWGraph(fswmm=fswmm)
        self.subc_ids_to_labels = {}
        self.labels_to_subc_ids = defaultdict(set)

    def dist(self, i1, i2):
        ''' 
        Returns the distance between the semantic land tags of two subcatchments 
        '''
        s1 = self.graph.subcatchmentids[int(i1)]
        s2 = self.graph.subcatchmentids[int(i2)]
        d1 = self.graph.landuse_percent[s1]
        d2 = self.graph.landuse_percent[s2]
        return sum(abs(d1.get(u,0) - d2.get(u,1)) for u in set.union(set(d1), set(d2)))

    def compute_subcatchment_clusters(self, distance_threshold=0.2):
        '''
        Compute clusters of subcatchments.
        
        Returns two dictionaries: {subc:label}, {label:{subc}}
        '''
        # Compute pairwise distances
        subc_ids = np.array(range(len(self.graph.subcatchmentids))).astype(int).reshape(-1,1)
        pdists = pairwise_distances(subc_ids,
                                    metric=self.dist,
                                    force_all_finite=False)

        # Cluster subcatchments in the graph based on their semantic land uses
        model = AgglomerativeClustering(distance_threshold=distance_threshold,
                                        n_clusters=None,
                                        affinity='precomputed',
                                        linkage='average')
        model.fit(pdists)
        
        subc_ids_to_labels = {self.graph.subcatchmentids[sid[0]] : c 
                              for sid,c in zip(subc_ids, model.labels_)}
        labels_to_subc_ids = defaultdict(set)
        for sid,c in subc_ids_to_labels.items():
            labels_to_subc_ids[c].add(sid)
        
        return subc_ids_to_labels, labels_to_subc_ids
        

    def generate_anomalies(self, n):
        
        anomalies = Anomalies()
        

class AnomalyGeneration:
    
    def __init__(self, G, Ahist):
        self.G, self.Ahist = G, Ahist

    def generate(self, n_gen):
        Asem = Anomalies(name='semantic')
        for i in range(n_gen):
            u_k = self.pick_landuse()
            v_k = self.pick_origin(u_k)
            tk_s, tk_e = self.pick_time_period(u_k)
            P_k = self.pick_phenomena(u_k)
            f = self.pick_flow(u_k)
            a_k = Anomaly(aid=i, origin=v_k, st=tk_s, et=tk_e, 
                          flow=f, phenomena=P_k, landuse=u_k)
            # print(a_k, f' caused by {u_k}')
            Asem.add_anomaly(a_k)
        return Asem

    def pick_landuse(self):
        U = list(SEMANTIC_LAND_USES)
        pr = np.array([sum(a_old.landuse == u_m for _,a_old in self.Ahist) for u_m in U])
        pr = pr / sum(pr)
        return np.random.choice(U, p=pr)
    
    def pick_origin(self, u_k):
        V = list(self.G)
        pr = np.array([self.G.get_attached_area(v, u_k) for v in V])
        pr = pr / sum(pr)
        return np.random.choice(V, p=pr)
    
    def pick_time_period(self, u_k):
        # Record start time and duration of anomalies caused by u_k
        Ts,Tdur = [], []
        for _,a_old in self.Ahist:
            if a_old.landuse == u_k:
                Ts.append(a_old.st)
                Tdur.append(a_old.et - a_old.st)
        tk_s = np.random.normal(np.mean(Ts), np.std(Ts))
        tdur = np.random.normal(np.mean(Tdur), np.std(Tdur))
        tk_e = tk_s + tdur
        return tk_s, tk_e
    
    def pick_phenomena(self, u_k):
        P_k = [a_old.phenomena for _,a_old in self.Ahist if a_old.landuse == u_k]
        return np.random.choice(P_k)
    
    def pick_flow(self, u_k):
        f = [a_old.flow for _,a_old in self.Ahist if a_old.landuse == u_k]
        return np.random.normal(np.mean(f), np.std(f))


if __name__ == '__main__':
    
    fhist = HISTORICAL_DATA[0]

    fhist_anomalies = utils.get_fall_historical_anomalies(fhist)
    Ahist = Anomalies(fhist_anomalies, name='historical')

    fsemantics_map = utils.get_fsemantic_map(fhist)
    smap, rsmap = utils.read_semantic_map(fsemantics_map)

    
    for fswmm in FSWMM:
        print(f'File: {fswmm}')
        G = SWGraph(fswmm)
        
        n_gen = G.number_of_nodes() * 3
        print(f'Generating {n_gen} anomalies')
        
        ag = AnomalyGeneration(G, Ahist)
        Asem = ag.generate(n_gen)

        fsemantic_anomalies = utils.get_fsemantic_anomalies(fswmm)
        Asem.dump(fsemantic_anomalies)
