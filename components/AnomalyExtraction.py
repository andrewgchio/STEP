import re
from datetime import datetime as dt, timedelta as td
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

import utils.utils as utils
from main.PARAMS import *
from network.StormwaterGraph import StormwaterGraph as SWGraph
from components.Anomalies import Anomalies, Anomaly, PHENOMENA
from network.NetworkAnalysis import propagation_time, entropy

def read_all_graphs():
    return {fswmmname : SWGraph(fswmm=fswmm, name=fswmmname)
            for fswmm, fswmmname in zip(FSWMM, FSWMM_NAME)}

class AnomalyExtraction:
    
    DURATION = lambda : int(max(np.random.normal(30,5),1))
    
    def __init__(self):
        self.k = 1
        self.Aext = defaultdict(lambda : Anomalies(name='historical'))
        self.Aext_all = Anomalies(name='historical')
        
        self.semantic_map = defaultdict(lambda : defaultdict(float))
        
        # Set threshold functions for each phenomena
        self.thresholds = {
            'turbidity'   : self._turbidity_thresh,
            'temperature' : self._temperature_thresh,
            'velocity'    : self._velocity_thresh,
            'ec'          : self._ec_thresh,
            'depth'       : self._depth_thresh
        }
        
    def extract_anomalies(self, Xhist, G, Aunif):
        '''
        Iterate through Xhist and construct new instances of anomalies
        '''
        pt = propagation_time(G, Aunif, full=False)
        
        # Preprocessing on Xhist
        Xhist['Time'] = Xhist['Time'].apply(lambda x : str(x)[:5])
        Xhist['Datetime'] = Xhist.apply(lambda x : dt.strptime(x['Date']+x['Time'], '%Y-%m-%d%H:%M'), axis=1)

        # Remove any non-numeric values
        num = re.compile(r'[^\d\.]')
        clean_num = lambda x : float(re.sub(num, '', str(x)) or -1) # Use -1 as a placeholder
        Xhist['Turbidity'] = Xhist['Turbidity'].apply(clean_num)
        Xhist['Water Temperature'] = Xhist['Water Temperature'].apply(clean_num)
        Xhist['Discharge Rate'] = Xhist['Discharge Rate'].apply(clean_num)
        Xhist['Electrical Conductivity'] = Xhist['Electrical Conductivity'].apply(clean_num)

        fproptime_cache = utils.get_fproptime_cache(G.fswmm, 'uniform')
        Xflow = pd.read_csv(fproptime_cache)
        
        for _,r in tqdm(Xhist.iterrows(), total=Xhist.shape[0]):
            for v_k, tk_s, tk_e, f_k in self.match_physics(Xflow, pt, r):
                P_k = self.find_anomalous_phenomena(r)
                if not P_k: # anomaly is not detectable with sensor
                    continue
                # Set the minimium flow to be 1.7, the average of the historical flows when too low
                # The argument is that the flow has decayed by the time it got to the node
                a_k = Anomaly(self.k, origin=v_k, st=tk_s, et=tk_e, flow=max(0.17,f_k), phenomena=P_k)
                self.Aext[G.name].add_anomaly(a_k)
                self.Aext_all.add_anomaly(a_k)
                self.k += 1

        self.update_semantic_map(G)
        
    def match_physics(self, Xflow, pt, r, tol=0.2):
        '''
        Return tuples (v_k, tk_s, tk_e, f_k) that most closely match anomalies 
        from Aunif
        
        Initially, this is assumed to be any of the upstream nodes
        '''
        v_j = r['Node']
        matches = []
        for v_i in nx.ancestors(G, v_j) | {v_j}:
            if pt[v_i,v_j] <= 1800:
                tk_s = (utils.seconds_since_midnight(r['Datetime']) - pt[v_i,v_j]) // 60
                tk_e = tk_s + AnomalyExtraction.DURATION()
                f_k = np.mean(Xflow[(Xflow['SrcID'] == v_i) & (Xflow['DstID'] == v_j)]['Flow'])
                diff = abs(f_k - r['Discharge Rate'])
                if diff < tol:
                    # matches.append((diff, v_i,tk_s,tk_e,f_k))
                    matches.append((v_i,tk_s,tk_e,f_k))
        return matches

    def find_anomalous_phenomena(self, r):
        '''
        Return the set of phenomena that were observed to be above threshold 
        values
        '''
        anomalous_phenomena = []
        if self.thresholds['turbidity'](r['Turbidity'], r['Datetime']):
            anomalous_phenomena.append('turbidity')
        if self.thresholds['temperature'](r['Water Temperature'], r['Datetime']):
            anomalous_phenomena.append('temperature')
        if self.thresholds['velocity'](r['Discharge Rate'], r['Datetime']):
            anomalous_phenomena.append('velocity')
        if self.thresholds['ec'](r['Electrical Conductivity'], r['Datetime']):
            anomalous_phenomena.append('ec')
        if self.thresholds['depth'](r['Discharge Rate'], r['Datetime']):
            anomalous_phenomena.append('depth')
        return anomalous_phenomena

        # print('datetime', dt)
        # print('turbidity', self.thresholds['turbidity'](r['Turbidity'], r['Datetime']))
        # print('temperature', self.thresholds['temperature'](r['Water Temperature'], r['Datetime']))
        # print('velocity', self.thresholds['velocity'](r['Discharge Rate'], r['Datetime']))
        # print('ec', self.thresholds['ec'](r['Electrical Conductivity'], r['Datetime']))
        # print('depth', self.thresholds['depth'](r['Discharge Rate'], r['Datetime']))
    
    def update_semantic_map(self, G):
        '''
        Updates the semantic map between semantic land uses and the types of 
        anomalies that they can create
        '''
        for _,a_ext in self.Aext[G.name]:
            landuses = G.get_landuses(a_ext.origin)
            for u,a in landuses.items():
                self.semantic_map[u][frozenset(a_ext.phenomena)] += a * entropy(landuses)
            
    def dump_semantic_map(self, fsemantic_map):
        '''
        Saves the semantic map to the file
        '''
        with open(fsemantic_map, 'w') as f:
            f.write('Landuse,PhenomenaSet,Weight\n')
            
            for u,anomaly_weights in self.semantic_map.items():
                for phenomena,weight in anomaly_weights.items():
                    ph = ';'.join(sorted(phenomena))
                    f.write(f'{u},{ph},{weight}\n')
    
    ############################################################################ 
    # Thresholding functions
    ############################################################################ 
    
    def _turbidity_thresh(self, x, dt):
        '''
        Return True if the turbidity value is outside the permitted threshold
        values at a datetime dt
        
        Unit: NTU
        '''
        return x >= 13 

    def _temperature_thresh(self, x, dt):
        '''
        Return True if the temperature value is outside the permitted threshold
        values at a datetime dt
        
        Unit: deg C
        '''
        return x >= 25.17

    def _velocity_thresh(self, x, dt):
        '''
        Return True if the depth value is outside the permitted threshold values
        at the datetime dt
        '''
        # TODO: this should really be through a simulation
        return x >= 0.18 # Currently the average flowrate reported
    
    def _ec_thresh(self, x, dt):
        '''
        Return True if the electric conductivity value is outside the permitted
        threshold values at a datetime dt
        
        Unit: uS/cm
        '''
        return x >= 3759
    
    def _depth_thresh(self, x, dt):
        '''
        Return True if the depth value is outside the permitted threshold values
        at the datetime dt
        '''
        # TODO: this should really be through a simulation
        return x >= 0.18 # Currently the average flowrate reported

if __name__ == '__main__':
    
    utils.setup_pandas()
    
    fpaths = {fswmmname : fswmm for fswmm,fswmmname in zip(FSWMM, FSWMM_NAME)}

    ae = AnomalyExtraction()
    for fhist in HISTORICAL_DATA:
        fhistsave = utils.get_fhistsave(fhist)
        for networkname, Xhist in pd.read_csv(fhistsave).groupby('Network'):
            fswmm = fpaths[networkname]
            
            G = SWGraph(fswmm=fswmm, name=networkname)
            
            funiform_anomalies = utils.get_funiform_anomalies(fswmm)
            Aunif = Anomalies(funiform_anomalies, name='uniform')
            
            # Iterate through Xhist, extract the anomaly, and cache it
            print(f'Extracting from {fswmm}')
            ae.extract_anomalies(Xhist, G, Aunif)
        
        for fswmm,fswmmname in zip(FSWMM, FSWMM_NAME):
            fsemantic_map = utils.get_fsemantic_map(fhist)
            smap,rsmap = utils.read_semantic_map(fsemantic_map)
            ae.Aext[fswmmname].augment_with_semantics(rsmap)
            
            fhistorical_anomalies = utils.get_fhistorical_anomalies(fswmm)
            ae.Aext[fswmmname].dump(fhistorical_anomalies)
            print(f'Saving to: {fhistorical_anomalies}')
        
        fsemantic_map = utils.get_fsemantic_map(fhist)
        ae.dump_semantic_map(fsemantic_map)

        fall_historical_anomalies = utils.get_fall_historical_anomalies(fhist)
        ae.Aext_all.dump(fall_historical_anomalies)

        print(f'Finished processing {fhist}')
    