import math
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime as dt, timedelta as td

import numpy as np
import pandas as pd

def setup_pandas():
    pd.set_option('display.max_rows',    5000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width',       1500)

class Memoize:
    def __init__(self, f):
        self._cache = {}
        self._f     = f
    
    def __call__(self, *args):
        if args in self._cache:
            return self._cache[args]
        ans = self._f(*args)
        self._cache[args] = ans
        return ans
    
    def clear_cache(self):
        self._cache.clear()

def dist(p1, p2):
    '''
    Return the distance between the two nodes
    '''
    return math.sqrt(dist2(p1,p2))

def dist2(p1, p2):
    '''
    Return the squared distance between the two nodes
    '''
    return (p1[0]-p2[0]) * (p1[0]-p2[0]) + (p1[1]-p2[1]) * (p1[1]-p2[1])

def minutes_since_midnight(x):
    timeobj = x.time()
    return 60*timeobj.hour + timeobj.minute

def seconds_since_midnight(x):
    timeobj = x.time()
    return 24*60*timeobj.hour + 60*timeobj.minute + timeobj.second

def read_semantic_map(fsemantics_map):
    smap = defaultdict(lambda : defaultdict(float))
    rsmap = defaultdict(lambda : defaultdict(float))
    for _,r in pd.read_csv(fsemantics_map).iterrows():
        P_k = frozenset(r['PhenomenaSet'].strip().split(';'))
        smap[r['Landuse']][P_k] += r['Weight']
        rsmap[P_k][r['Landuse']] += r['Weight']
    return smap,rsmap

################################################################################
# IO Utils
################################################################################

def add_to_fname(fname, directory=None, append='_edited', suffix=None, rename=None):
    fpath = Path(fname).resolve()
    newname = rename if rename else fpath.stem
    if directory is None:
        directory = fpath.parent
    else:
        directory = (fpath.parent / directory).resolve()
    if suffix == None:
        return directory / (newname + append + fpath.suffix)
    else:
        return directory / (newname + append + '.' + suffix)

def get_fosm(fswmm):
    return add_to_fname(fswmm, append='_osm', suffix='osm')

def get_ftags(fswmm):
    return add_to_fname(fswmm, append='_tags', suffix='csv')

def get_fsubctags(fswmm):
    return add_to_fname(fswmm, append='_subctags', suffix='csv')

def get_fnodetags(fswmm):
    return add_to_fname(fswmm, append='_nodetags', suffix='csv')

def get_fcoords(fswmm):
    return add_to_fname(fswmm, append='_coords', suffix='csv')

def get_fedges(fswmm):
    return add_to_fname(fswmm, append='_edges', suffix='csv')

# Network Property Caches
def get_fbranchingcomplexity(fswmm):
    return add_to_fname(fswmm, append='_bc', suffix='csv')
def get_fsemanticentropyindex(fswmm):
    return add_to_fname(fswmm, append='_sei', suffix='csv')
def get_fdensity(fswmm):
    return add_to_fname(fswmm, append='_den', suffix='csv')
def get_fbetweennesscentrality(fswmm):
    return add_to_fname(fswmm, append='_btn', suffix='csv')

def get_fgeojson(fswmm, item):
    return add_to_fname(fswmm, append=f'_{item}_geojson', suffix='geojson')

# Anomalies
def get_funiform_anomalies(fswmm):
    return add_to_fname(fswmm, append='_uniform_anomalies_v2', suffix='csv')
def get_fhistorical_anomalies(fhist):
    return add_to_fname(fhist, append='_historical_anomalies', suffix='csv')
def get_fall_historical_anomalies(fhist):
    return add_to_fname(fhist, directory='../data/cache/', append=f'_all_historical_anomalies', suffix='csv')
def get_fsemantic_anomalies(fswmm):
    return add_to_fname(fswmm, append='_semantic_anomalies', suffix='csv')

def get_fsemantic_map(fhist):
    return add_to_fname(fhist, append='_semantic_map', suffix='csv')

def get_fproptime(fswmm, anomalies_type):
    return add_to_fname(fswmm, append=f'_proptime_{anomalies_type}', suffix='csv')

def get_fproptime_cache(fswmm, anomalies_type, n1=None, n2=None):
    if n1 is None and n2 is None:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_proptime_cache_{anomalies_type}', suffix='csv')
    else:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_proptime_cache_{anomalies_type}_{n1}_{n2}', suffix='csv')
    
def get_fsimu_cache(fswmm, anomalies_type, n1=None, n2=None):
    if n1 is None and n2 is None:
        return add_to_fname(fswmm, directory='../../data/simulations/', append=f'_simu_cache_{anomalies_type}', suffix='csv')
    else:
        return add_to_fname(fswmm, directory='../../data/simulations/', append=f'_simu_cache_{anomalies_type}_{n1}_{n2}', suffix='csv')
    
################################################################################
# Files for Baselines
################################################################################

# Greedy Heuristic Placement
def get_fgreedy_coverage_cache(fswmm, Aname):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_greedy_coverage_cache_{Aname}', suffix='csv')
def get_fgreedy_priority_cache(fswmm, Aname):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_greedy_priority_cache_{Aname}', suffix='csv')
def get_fgreedy_traceability_cache(fswmm, Aname):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_greedy_traceability_cache_{Aname}', suffix='csv')
def get_fgreedy_ph_traceability_cache(fswmm, Aname):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_greedy_ph_traceability_cache_{Aname}', suffix='csv')
def get_fgreedy_se_traceability_cache(fswmm, Aname):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_greedy_se_traceability_cache_{Aname}', suffix='csv')

def get_fgreedy_radius_coverage_cache(fswmm):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_greedy_radius_coverage', suffix='csv')
def get_fgreedy_centrality_coverage_cache(fswmm):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_greedy_centrality_coverage', suffix='csv')

    
# Genetic Algorithm Placement
def get_fgenetic_coverage_cache(fswmm, Aname, B=None):
    if B is None:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_genetic_coverage_cache_{Aname}', suffix='csv')
    else:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_genetic_coverage_cache_{Aname}_{B}', suffix='csv')
def get_fgenetic_priority_cache(fswmm, Aname, B=None):
    if B is None:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_genetic_priority_cache_{Aname}', suffix='csv')
    else:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_genetic_priority_cache_{Aname}_{B}', suffix='csv')
def get_fgenetic_traceability_cache(fswmm, Aname, B=None):
    if B is None:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_genetic_traceability_cache_{Aname}', suffix='csv')
    else:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_genetic_traceability_cache_{Aname}_{B}', suffix='csv')
def get_fgenetic_ph_traceability_cache(fswmm, Aname, B=None):
    if B is None:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_genetic_ph_traceability_cache_{Aname}', suffix='csv')
    else:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_genetic_ph_traceability_cache_{Aname}_{B}', suffix='csv')
def get_fgenetic_se_traceability_cache(fswmm, Aname, B=None):
    if B is None:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_genetic_se_traceability_cache_{Aname}', suffix='csv')
    else:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_genetic_se_traceability_cache_{Aname}_{B}', suffix='csv')
    
# Linear Program Optimization Placement
def get_flp_coverage_cache(fswmm, Aname, B=None):
    if B is None:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_lp_coverage_cache_{Aname}', suffix='csv')
    else:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_lp_coverage_cache_{Aname}_{B}', suffix='csv')
def get_flp_traceability_cache(fswmm, Aname, B=None):
    if B is None:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_lp_traceability_cache_{Aname}', suffix='csv')
    else:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_lp_traceability_cache_{Aname}_{B}', suffix='csv')
def get_flp_ph_traceability_cache(fswmm, Aname, B=None):
    if B is None:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_lp_ph_traceability_cache_{Aname}', suffix='csv')
    else:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_lp_ph_traceability_cache_{Aname}_{B}', suffix='csv')
def get_flp_se_traceability_cache(fswmm, Aname, B=None):
    if B is None:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_lp_se_traceability_cache_{Aname}', suffix='csv')
    else:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_lp_se_traceability_cache_{Aname}_{B}', suffix='csv')
def get_flp_coverage_log(fswmm, Aname, B=None):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_lp_coverage_cache_{Aname}_{B}_log', suffix='csv')
def get_flp_traceability_log(fswmm, Aname, B=None):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_lp_traceability_cache_{Aname}_{B}_log', suffix='csv')
def get_flp_ph_traceability_log(fswmm, Aname, B=None):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_lp_ph_traceability_cache_{Aname}_{B}_log', suffix='csv')
def get_flp_se_traceability_log(fswmm, Aname, B=None):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_lp_se_traceability_cache_{Aname}_{B}_log', suffix='csv')
   

################################################################################
# Cached heuristic values
################################################################################

def get_fcoverage_heuristic_cache(fswmm, Aname):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_coverage_heuristic_cache_{Aname}', suffix='csv')
def get_fph_traceability_heuristic_cache(fswmm, Aname):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_ph_traceability_heuristic_cache_{Aname}', suffix='csv')
def get_fse_traceability_heuristic_cache(fswmm, Aname):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_se_traceability_heuristic_cache_{Aname}', suffix='csv')
def get_fbranching_complexity_heuristic_cache(fswmm, Aname):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_branching_complexity_heuristic_cache_{Aname}', suffix='csv')
def get_fsemantic_entropy_heuristic_cache(fswmm, Aname):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_semantic_entropy_heuristic_cache_{Aname}', suffix='csv')
def get_fbetweenness_centrality_heuristic_cache(fswmm, Aname):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_betweenness_centrality_heuristic_cache_{Aname}', suffix='csv')
def get_fpropagation_time_heuristic_cache(fswmm, Aname):
    return add_to_fname(fswmm, directory='../data/cache/', append=f'_propagation_time_heuristic_cache_{Aname}', suffix='csv')

################################################################################
# Historical data
################################################################################

def get_fhistsave(fhist):
    return add_to_fname(fhist, append=f'_processed', suffix='csv')

################################################################################
# STEP Approach
################################################################################

def get_fstep_betweenness_centrality_partitions(fswmm, Aname):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_step_btn_{Aname}', suffix='csv')
def get_fstep_branching_complexity_partitions(fswmm):
    return add_to_fname(fswmm, directory='../../data/cache/', append=f'_step_bc', suffix='csv')

def get_fstep(fswmm, Aname, n=None):
    if n is None:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_step_{Aname}', suffix='csv')
    else:
        return add_to_fname(fswmm, directory='../../data/cache/', append=f'_step_{Aname}_{n}', suffix='csv')

################################################################################
# Merged Plots
################################################################################

def get_fcoverage_vs_budget_merged(fswmm_dummy, Aname):
    return add_to_fname(fswmm_dummy, directory='../../data/figures/', append=f'_cov_vs_budget_{Aname}', suffix='eps')
def get_fanomaly_coverage_vs_budget_merged(fswmm_dummy, Aname):
    return add_to_fname(fswmm_dummy, directory='../../data/figures/', append=f'_anom_cov_vs_budget_{Aname}', suffix='eps')
def get_ftraceability_vs_budget_merged(fswmm_dummy, Aname):
    return add_to_fname(fswmm_dummy, directory='../../data/figures/', append=f'_tr_vs_budget_{Aname}', suffix='eps')

################################################################################
# Adding a human in the loop
################################################################################

def get_fhuman(fswmm, anomalies, B):
    return add_to_fname(fswmm, directory='../../data/cache', append=f'_human_{anomalies}_{B}', suffix='csv')






