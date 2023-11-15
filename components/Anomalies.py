import random 

import numpy as np
import pandas as pd

import utils.utils as utils

PHENOMENA = [
    'turbidity',
    'temperature',
    'velocity',
    'ec',
    'depth'
]

class Anomaly:

    def __init__(self, aid, origin, st, et, flow, phenomena, landuse=None):
        '''
        Initialize an anomaly with a given: 
            aid: id of the anomaly
            origin: origin node of the anomaly
            st,et: start and end time of the anomaly
            flow: flow amount for the anomaly
            phenomena: the set of phenomena produced by the sensor
            landuse: the landuse producing the anomaly 
        '''
        self.aid = aid
        self.origin = origin
        self.st, self.et = st, et
        self.flow = abs(flow)
        self.phenomena = phenomena
        self.landuse = landuse

    def __str__(self):
        return f'''Anomaly {self.aid} (origin={self.origin}, st,et={self.st} - {self.et}, phenomena={self.phenomena}, flow={self.flow}, landuse={self.landuse}) '''

    def dump(self):
        phenomena = ';'.join(self.phenomena)
        return f'{self.aid},{self.origin},{self.st},{self.et},{self.flow},{self.landuse},{phenomena}'

class Anomalies:

    MIN_TIME = 0
    MAX_TIME = 1440
    DURATION = lambda : int(max(np.random.normal(30,5),1))
    
    ITEMS = {}
    
    def __init__(self, fanomalies=None, name=None):
        self.data = {}
        self.name = name
        
        Anomalies.ITEMS[self.name] = self
        
        if fanomalies is not None:
            self.load_anomalies(fanomalies)
    
    def load_anomalies(self, fname):
        Xanomalies = pd.read_csv(fname)
        for _,r in Xanomalies.iterrows():
            phenomena = {p for p in r['Phenomena'].strip().split(';')}
            if 'Landuse' in r:
                self.data[r['AnomalyID']] = Anomaly(r['AnomalyID'],
                                         r['OriginNodeID'],
                                         r['StartTime'],
                                         r['EndTime'],
                                         r['Flow'],
                                         phenomena,
                                         r['Landuse'])
            else:
                self.data[r['AnomalyID']] = Anomaly(r['AnomalyID'],
                                         r['OriginNodeID'],
                                         r['StartTime'],
                                         r['EndTime'],
                                         r['Flow'],
                                         phenomena)
    

    def __str__(self):
        return '\n'.join(map(str, self.data.values()))
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data.items())
    
    def __getitem__(self, i):
        return self.data[i]
    
    def get(self, v):
        '''
        Return the set of anomalies that originate at node v
        '''
        return {i:x for i,x in self.data.items() if x.origin == v}
    
    def augment_with_semantics(self, rsmap):
        '''
        Assign a landuse "cause" to each anomaly
        '''
        for a_k in self.data.values():
            P_k = frozenset(a_k.phenomena)
            U = list(rsmap[P_k])
            pr = np.array([rsmap[P_k][u] for u in U])
            pr /= np.sum(pr) # normalize values
            a_k.landuse = np.random.choice(U, p=pr)

    def select_random_origin(self, nodes):
        return random.choice(nodes)

    def select_random_time(self):
        st = random.randint(Anomalies.MIN_TIME, Anomalies.MAX_TIME)
        dur = Anomalies.DURATION()
        et = st+dur
        return st,et

    def sample_phenomena(self):
        phenomena_combinations = [
            {'turbidity', 'ec'},
            {'turbidity', 'ec', 'temperature'},
            {'temperature'},
            {'depth', 'velocity'}
        ]
        return random.choice(phenomena_combinations)

    def sample_flow(self):
        return max(0.1, np.random.normal(0.2, 0.1))

    def add_anomaly(self, a_k):
        self.data[a_k.aid] = a_k

    def add_uniform_anomalies(self, G, flowrange, n):
        k = 0
        for v in G.nodes():
            for f in flowrange:
                for _ in range(n):
                    st,et = self.select_random_time()
                    phenomena = self.sample_phenomena()

                    anomaly = Anomaly(aid=k, 
                                      origin=v, 
                                      st=st, et=et, 
                                      flow=f,
                                      phenomena=phenomena)
                    self.data[k] = anomaly
                    
                    k += 1
    
    def dump(self, fname):
        with open(fname, 'w') as f:
            f.write('AnomalyID,OriginNodeID,StartTime,EndTime,Flow,Landuse,Phenomena\n')
            for a in self.data.values():
                f.write(a.dump())
                f.write('\n')
        
# memoized functions
@utils.Memoize
def getAnomaliesAt(Aname, v):
    return Anomalies.ITEMS[Aname].get(v)



