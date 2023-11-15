# StormwaterGraph.py

import re
import json
# from os.path import exists
from collections import defaultdict
from itertools import count

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import BASE_COLORS as MCOLORS

# import hymo
from hymo import hymo
from pyproj import CRS
from shapely import Polygon, Point
from pyswmm import Simulation, Nodes, Links, Subcatchments

import utils.utils as utils
from main.PARAMS import SEMANTIC_LAND_USES
from utils.CoordinateSystem import CoordinateSystem

class StormwaterGraph(nx.DiGraph):
    
    ############################################################################ 
    # Constructor/"Destructor"
    ############################################################################ 
    
    def __init__(self, fswmm=None, ftags=None, name=None):
        '''
        Construct the Stormwater graph object and initialize instance variables
        '''

        # Initialize digraph
        nx.DiGraph.__init__(self)
        
        self.name = name
        self.fswmm = fswmm
        self.ftags = ftags
        
        if fswmm is not None:
            self.load_swmm()

        if ftags is not None:
            self.load_tags()
        
    def load_swmm(self):
        '''
        Load a SWMM input file
        '''

        ######################################################################## 
        # Open/Parse SWMM file
        ######################################################################## 

        # Open SWMM file to read some parts directly
        self.fhymo = hymo.SWMMInpFile(self.fswmm)
        
        # Open SWMM file through pyswmm
        self.sim = Simulation(self.fswmm); print()
        
        # Read junction coordinates
        coords_map = self._read_coordinates()
        
        # Read subcatchment polygons 
        polygons_map = self._read_polygons()
        
        # Read outlet junctions for each subcatchments
        self.subc_outlets = self._read_subcatchment_outlets()

        # Read landuse percents and areas
        landuse_percent_map = self._read_landuse_percents()
        
        ######################################################################## 
        # Initialize digraph nodes and edges
        ######################################################################## 

        # Set nodes
        crs = CoordinateSystem(source=CoordinateSystem.CRS_2230,
                               target=CoordinateSystem.CRS_3857)
        for n in Nodes(self.sim):
            self.add_node(n.nodeid, 
                          pos=coords_map[n.nodeid], 
                          latlong=crs(*coords_map[n.nodeid]),
                          data=n, 
                          tags=defaultdict(int))

        # Set edges
        for l in Links(self.sim):
            self.add_edge(*l.connections, data=l)

        ######################################################################## 
        # Initialize subcatchments
        ######################################################################## 
        
        # Set subcatchments
        self.subc = {}
        for s in Subcatchments(self.sim):
            sid = s.subcatchmentid
            
            # Remove UnAcc subcatchment
            if 'UnAcc' in self.subc_outlets and \
                sid in self.subc_outlets['UnAcc']:
                continue

            
            self.subc[sid] = {
                    'data' : s, 
                    'polygon' : polygons_map[sid],
                    'area' : s.area,
                    'landuse_percent' : landuse_percent_map[sid],
                    'landuse_area' : {u : p * s.area 
                                  for u,p in landuse_percent_map[sid].items()},
                    'landuse_weight' : sum(p * s.area * SEMANTIC_LAND_USES[u]
                                  for u,p in landuse_percent_map[sid].items()),
                    'tags' : []}

        # Remove UnAcc subcatchment
        if 'UnAcc' in self.subc_outlets:
            del self.subc_outlets['UnAcc']

    def load_tags(self):
        pass

    def close_sim(self):
        self.sim.close()

    ############################################################################ 
    # Graph Queries
    ############################################################################ 

    def find_containing_subcatchment(self, xy):
        '''
        Return the subcatchment that contains the provided coordinate point

        TODO: Can be improved by using a more efficient data structure (like a
        doubly connected edge list)
        '''
        p = Point(xy)
        for sid,subc in self.subc.items():
            if subc['polygon'].contains(p):
                return sid
        return None

    def find_nearby_nodes(self, xy, radius):
        '''
        Return the set of nodes that are within the given radius of (x,y)

        TODO: Can be improved by using a more efficient data structure (like a 
        doubly connected edge list)
        '''
        nids = set()
        for nid in self.nodes:
            if utils.dist2(xy, self.nodes[nid]['pos']) < radius * radius:
                nids.add(nid)
        return nids

    def find_nodes_within_radius(self, nid, radius):
        xy = self.nodes[nid]['pos']
        return self.find_nearby_nodes(xy,radius)

    def find_attached_subcatchments(self, nid):
        '''
        Return the subcatchment ids associated with the node id
        '''
        # Attached subcatchment already exists
        if nid in self.subc_outlets:
            return self.subc_outlets[nid]

        # Find the upstream attached subcatchment
        nids = {nid}
        outlets = set()
        while nids:
            preds = set()
            for nid in nids:
                for pred in self.predecessors(nid):
                    preds.add(pred)
                    if pred in self.subc_outlets:
                        outlets |= self.subc_outlets[pred]
        
            if outlets: # At least one attached subcatchment
                return outlets
        
            nids = preds

        return set() # Nothing found upstream

    def get_landuses(self, nid):
        '''
        Return the mapping of landuses associated with the subcatchments going
        to the node nid
        '''
        total_landuse_area = defaultdict(float)
        for sid in self.find_attached_subcatchments(nid):
            for u,a in self.subc[sid]['landuse_area'].items():
                total_landuse_area[u] += a
        return total_landuse_area

    def get_attached_area(self, nid, landuse=None):
        '''
        Return the total area attached to a node. If landuse is provided, then
        return the area of the specific landuse. 
        '''
        total_landuse_area = self.get_landuses(nid)
        if landuse:
            return total_landuse_area[landuse]
        else:
            return sum(total_landuse_area.values())
        
#     def find_enclosed_tags(self, sid):
#         '''
#         Return the semantic tags that are within the subcatchment. 
#         '''
#         return self.subc[sid]['tags']
#
#     def find_tags_near_node(self, nid):
#         '''
#         Return the semantic tags that are near node nid
#         '''
#         return self.nodes[nid]['tags']
#
#     def collapse_graph(self, show=False):
#         '''
#         Return the collapsed graph that combines nodes in a chain. The new graph
#         should only contain a shell of the original. 
#
#         TODO: Check that constructed graph does not contain G's metadata
#         '''
#         # Copy nodes and edges of G
#         Gp = nx.DiGraph()
#         Gp.add_nodes_from(self.nodes)
#         Gp.add_edges_from(self.edges)
#
#         if show:
#             fig,ax = plt.subplots(2,1)
#             nx.draw(Gp, ax=ax[0], with_labels=True)
#             ax[0].set_title('Original')
#
#         uncollapsed = [v for v in Gp.nodes]
#         while uncollapsed:
#             node = uncollapsed.pop(-1)
#
#             if Gp.in_degree(node) == 1: # Collapse this node
#
#                 pred = next(iter(Gp.predecessors(node)))
#                 for succ in Gp.neighbors(node):
#                     Gp.add_edge(pred, succ)
#                 Gp.remove_node(node)
#
#         if show:
#             nx.draw(Gp, ax=ax[1], with_labels=True)
#             ax[1].set_title('Collapsed')
#             plt.show()
#
#         return Gp
#
    def get_bounding_box(self):
        coords = self.fhymo.coordinates.reset_index()
        Xmin = np.min(coords['X_Coord'])
        Ymin = np.min(coords['Y_Coord'])
        Xmax = np.max(coords['X_Coord'])
        Ymax = np.max(coords['Y_Coord'])

        return Xmin, Ymin, Xmax, Ymax


    ############################################################################ 
    # I/O Queries
    ############################################################################ 

    def visualize(self, placement=None, metric=None, subcatchments=False, save=None, show=False):
        '''
        Read a csv file of [Node, Sensor] and visualize the placement

        :param placement: a file containing node,sensor assignments
        :param metric: a map of node to float; visualize the value at nodes
        :param subcatchments: True if subcatchment outlines should be drawn
        :param save: a file to save the visualization to, default None
        :param show: True if the visualization should be shown
        '''

        fig,ax = plt.subplots()

        if subcatchments:
            for s in self.subc.values():
                plt.plot(*s['polygon'].exterior.xy, color='bisque', zorder=1)

            # Sample subcatchments to visualize, only for Newport database
            # for sid in ['c330', 'c503']:
            #     subc = self.subcatchments[sid]
            #     plt.plot(*subc['polygon'].exterior.xy, color='red', zorder=1)
        
        if metric: 
            colors = [np.log(metric[v]) for v in self.nodes]
            Vdraw = nx.draw_networkx_nodes(self,nx.get_node_attributes(self,'pos'),
                                           node_size=5, 
                                           node_color=colors, 
                                           ax=ax)
            Edraw = nx.draw_networkx_edges(self,nx.get_node_attributes(self,'pos'),
                                           arrowsize=5,
                                           ax=ax)
            plt.sci(Vdraw)
            plt.colorbar() # Display metric
        
        else:
            nx.draw(self, nx.get_node_attributes(self,'pos'), with_labels=False,
                    node_size=5, node_color='blue', 
                    arrowsize=5,
                    ax=ax)

        if type(placement) is str:
            placement = pd.read_csv(placement, 
                                    names=['Node', 'Sensor'],
                                    dtype=str)

            for (_,gp),c in zip(placement.groupby('Sensor'), MCOLORS):
                subg = self.subgraph(gp['Node'])
                nx.draw(subg, nx.get_node_attributes(subg,'pos'), 
                        with_labels=False,
                        node_size=5, node_color=c, ax=ax)

        if type(placement) is dict: 
            subg = self.subgraph(placement.keys())
            nx.draw(subg, nx.get_node_attributes(subg,'pos'), 
                    with_labels=False,
                    node_size=5, node_color='red', ax=ax)
        
        if type(placement) is list:
            nx.draw_networkx_nodes(self,nx.get_node_attributes(self,'pos'),
                                           nodelist=placement,
                                           node_size=5, 
                                           node_color='red', 
                                           ax=ax)




        if save:
            fig.savefig(save, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def print_graph_size_info(self):
        nv,ne = self.number_of_nodes(), self.number_of_edges()
        ns = len(self.subc)
        area = sum(s['area'] for s in self.subc.values()) * 0.00405 # Acre to km2
        print(f'Graph with {nv} nodes, {ne} edges, {ns} subc, {area} km2 area')
    
    def print_graph_semantics(self, keys='all'):
        semantics = defaultdict(float)
        for subc in self.subc.values():
            for u,a in subc['landuse_area'].items():
                if keys == 'all' or (type(keys) is set and u in keys):
                    semantics[u] += a
                else:
                    semantics['OTHER'] += a
        
        # Print out requested keys
        print('Semantics breakdown:')
        for u in SEMANTIC_LAND_USES:
            print(f'  {u} : {semantics[u]}')



        
        
#
#     def augment_with_semantics(self, stags, cache=None):
#         # Set flag
#         self.AUGMENTED_WITH_SEMANTICS = True
#
#         # Create subcatchments-tags cache if not exists
#         if not exists(cache[0]):
#             print('Starting subcatchment tag semantics...')
#             for _,r in tqdm(stags.data.iterrows(), total=stags.data.shape[0]):
#                 sid = self.find_containing_subcatchment((r['x'],r['y']))
#                 if sid: # xy exists
#                     self.subc[sid]['tags'].append(r['tagkey'])
#                     print(f'{sid} : tag(' + self.subc[sid]['tags'] + ')')
#                 # else:
#                 #     print('StormwaterGraph::augment_with_semantics: xy not found', r['x'], r['y'])
#
#             self._cache_semantics_for_subcatchments(cache[0])
#
#         else:
#             Xtags = pd.read_csv(cache[0], names=['sid','tagkey'])
#             print('Reading subcatchment tags')
#             for _,r in tqdm(Xtags.iterrows(), total=Xtags.shape[0]):
#                 self.subc[r['sid']]['tags'].append(r['tagkey'])
#
#         if not exists(cache[1]):
#             print('Starting node tag semantics...')
#             for _,r in tqdm(stags.data.iterrows(), total=stags.data.shape[0]):
#                 nids = self.find_nearby_nodes((r['x'],r['y']), radius=10000)
#                 for nid in nids:
#                     self.nodes[nid]['tags'][r['tagkey']] += 1
#             self._cache_semantics_for_nodes(cache[1])
#
#         else:
#             Xtags = pd.read_csv(cache[1], names=['nid','tagkey','count'])
#             print('Reading node tags')
#             for _,r in tqdm(Xtags.iterrows(), total=Xtags.shape[0]):
#                 self.nodes[r['nid']]['tags'][r['tagkey']] = r['count']
#
#     def _cache_semantics_for_subcatchments(self, cache):
#         with open(cache, 'w') as f:
#             for sid,d in self.subc.items():
#                 for tag in d['tags']:
#                     f.write(f'{sid},{tag}\n')
#
#     def _cache_semantics_for_nodes(self, cache):
#         with open(cache, 'w') as f:
#             for nid in self.nodes:
#                 for tag,count in self.nodes[nid]['tags'].items():
#                     f.write(f'{nid},{tag},{count}\n')
#
    def write_subcatchments_geojson(self, fname):
        '''
        Write a GeoJSON object for subcatchments
        '''

        crs = CoordinateSystem(source=CoordinateSystem.CRS_2230, 
                               target=CoordinateSystem.CRS_4326)

        # While easier to write into a dict object and output, such a file will
        # take a long time to parse. 
        with open(fname, 'w') as f:

            f.write('{"type":"FeatureCollection","features":[\n')

            entries = []
            for sid,s in self.subc.items():

                data = {}
                data['type'] = 'Feature'
                data['id']   = sid
                data['properties'] = {'name' : sid, 
                                      'network' : self.network_name, 
                                      'area' : s['data'].area}
                data['geometry'] = {'type' : 'Polygon', 'coordinates' : [[]]}

                for px,py in zip(*s['polygon'].exterior.xy):
                    py,px = crs(px,py) # Need in this order
                    data['geometry']['coordinates'][0].append([px,py])

                json_str = json.dumps(data)
                entries.append(json_str)

            f.write(',\n'.join(entries))
            f.write('\n]}')

    def write_conduits_geojson(self, fname):
        '''
        Write a GeoJSON object for conduitsgg
        '''
        crs = CoordinateSystem(source=CoordinateSystem.CRS_2230, 
                               target=CoordinateSystem.CRS_4326)

        # While easier to write into a dict object and output, such a file will
        # take a long time to parse. 
        with open(fname, 'w') as f:

            f.write('{"type":"FeatureCollection","features":[\n')

            entries = []
            for eid,e in self.edges().items():
            # for sid,s in self.subc.items():

                data = {}
                data['type'] = 'Feature'
                data['id']   = eid

                j1,j2 = e['data'].connections
                (j1x,j1y),(j2x,j2y) = self.nodes[j1]['pos'], self.nodes[j2]['pos']
                # Compute length using same metric as in 'pos'
                data['properties'] = {'connection' : eid, 
                                      'network' : self.network_name, 
                                      'length' : utils.dist((j1x,j1y),(j2x,j2y))}

                # But use coordinates after translation
                (j1y,j2y),(j1x,j2x) = crs((j1x,j2x),(j1y,j2y)) # Need in this order
                data['geometry'] = {'type' : 'LineString', 
                                    'coordinates' : [ [j1x,j1y], [j2x,j2y] ]}

                json_str = json.dumps(data)
                entries.append(json_str)

            f.write(',\n'.join(entries))
            f.write('\n]}')

    ############################################################################ 
    # I/O Helper Utilities
    ############################################################################ 

    def _read_coordinates(self):
        '''
        Read and return a map of node ids to coordinates
        '''
        return {r['Node'] : (r['X_Coord'], r['Y_Coord'])
                for _,r in self.fhymo.coordinates.reset_index().iterrows()}

    def _read_polygons(self):
        '''
        Read and return a map of subcatchment ids to polygons
        '''
        return {s : Polygon(pts[['X_Coord', 'Y_Coord']]) 
                for s,pts in self.fhymo.polygons.reset_index()
                                                .groupby('Subcatchment')}

    def _read_subcatchment_outlets(self):
        '''
        Read and return a map of subcatchment ids to outlet node ids
        Read and return a map of node ids to associated subcatchments
        '''
        return {r['Outlet'] : r['Name']
                for _,r in self.fhymo.subcatchments.reset_index()
                                        .groupby('Outlet')
                                        .agg({'Name' : set})
                                        .reset_index()
                                        .iterrows()
                }

    def _read_landuse_percents(self):
        '''
        Read and return a map of landuses to their percents
        '''
        landuse_percents = defaultdict(lambda : defaultdict(float))

        # Status flag of file contents
        cov_status = 0 # 0 = not yet reached, 1 = reached, 2 = past

        # Pattern to match a section
        section_pat = re.compile(r'\[\w*\]')

        with open(self.fswmm) as f:

            for line in f:

                # Determine if you are in time series section
                # 0 = not yet reached, 1 = reached, 2 = past section
                if '[COVERAGES]' in line:
                    cov_status = 1
                elif cov_status == 1 and re.match(section_pat, line):
                    cov_status = 2

                # Read the line in the coverage section
                if cov_status == 1:
                    if line.startswith(';;'):
                        continue
                    splitline = line.strip().split()
                    if len(splitline) != 3:
                        continue

                    sid, landuse, percent = splitline
                    landuse_percents[sid][landuse] = float(percent)/100

        # Now, we combine them into buckets as necessary
        bucket_landuse_percents = defaultdict(lambda : defaultdict(float))
        for sid,landuses in landuse_percents.items():
            for landuse,percent in landuses.items():
                if landuse in SEMANTIC_LAND_USES:
                    bucket_landuse_percents[sid][landuse] += percent
                else:
                    bucket_landuse_percents[sid]['OTHER'] += percent
        return bucket_landuse_percents

#     def save_coords(self, fcoords):
#         with open(fcoords,'w') as f:
#             f.write('V,X,Y\n')
#             for nid in self.nodes:
#                 X,Y = self.nodes[nid]['pos']
#                 f.write(f'{nid},{X},{Y}\n')
#
#     def save_edges(self, fedges):
#         with open(fedges,'w') as f:
#             f.write('S,D\n')
#             for src in self.nodes:
#                 for dst in self.successors(src):
#                     f.write(f'{src},{dst}\n')
#
#
#     def save_branching_complexity(self, fbc):
#         # Gbc = self.collapse_graph()
#         bc = NetworkAnalysis.branching_complexity(self, method='branching')
#         with open(fbc,'w') as f:
#             f.write('node,bc\n')
#             for nid,v in bc.items():
#                 f.write(f'{nid},{v}\n')
#
#     def save_semantic_entropy_index(self, fsei):
#         sei = NetworkAnalysis.semantic_entropy_index(self)
#         with open(fsei,'w') as f:
#             f.write('node,sei\n')
#             for nid,v in sei.items():
#                 f.write(f'{nid},{v}\n')
#
#
#     def save_node_density(self, fden):
#         den = NetworkAnalysis.node_density(self)
#         with open(fden,'w') as f:
#             f.write('node,den\n')
#             for nid,v in den.items():
#                 f.write(f'{nid},{v}\n')
#
#     def save_betweenness_centrality(self, fbtn):
#         btn = NetworkAnalysis.betweenness_centrality(self)
#         with open(fbtn,'w') as f:
#             f.write('node,btn\n')
#             for nid,v in btn.items():
#                 f.write(f'{nid},{v}\n')
#
# def print_network_analysis_metrics(G, collapse=True):
#         Gbc = G.collapse_graph() if collapse else G
#
#         # Compute and print Branching Complexity
#         for bm in ['branching', 'strahler', 'subtree']:
#
#             # Compute complexity
#             complexity = NetworkAnalysis.branching_complexity(Gbc, method=bm)
#
#             # Consider nodes that are not only root nodes
#             bc_values = list(filter(lambda x : x > 1, complexity.values()))
#
#             # Print complexity value
#             str_collapse = ' (collapsed)' if collapse else ''
#             print(f'Mean for {bm}{str_collapse}:', np.mean(bc_values))
#
#         # Compute and print Semantic Entropy Index
#         sei = NetworkAnalysis.semantic_entropy_index(G)
#         print(f'Mean for semantic entropy:', np.mean(list(sei.values())))
#
#         # Compute and print Node Density
#         density = NetworkAnalysis.node_density(G)
#         print(f'Mean for node density:', np.mean(list(density.values())))
#
#         # Compute and print Reachability  
#         betweenness = NetworkAnalysis.betweenness_centrality(G)
#         print(f'Mean for betweenness_centrality:', np.mean(list(betweenness.values())))
#
#         # Compute and print Propagation Time
#         proptime = NetworkAnalysis.propagation_time(G)
#         print(f'Mean for propagation time:', np.mean(list(proptime.values())))
#
#

def upstream_subgraphs(G, X_v):
    Gup = []

    Gsplit = G.copy()
    
    # Split Gsub by X_v
    for v in nx.topological_sort(G):
        if v not in X_v:
            continue

        Vup = nx.ancestors(Gsplit, v) | {v}
        
        Gpred = nx.subgraph(Gsplit, Vup)
        Gsplit = nx.subgraph(Gsplit, [node for node in Gsplit.nodes if node not in Vup])
        
        # print(Gpred.number_of_nodes(), '+', Gsplit.number_of_nodes(), '=', Gsplit.number_of_nodes(), '?')
        
        # Gup.append(Gpred)
        yield Gpred
    
    if Gsplit.number_of_nodes() != 0:
        # Gup.append(Gsplit)
        yield Gsplit
    
    # assert G.number_of_nodes() == sum(g.number_of_nodes() for g in Gup), \
    #         f'Total number of nodes is not the same {G.number_of_nodes()} != {sum(g.number_of_nodes() for g in Gup)}'
                
    # return Gup

def upstream_subgraphs_count(G, X_v):
    Gup_sizes = []
    Gsplit = G.copy()
    
    for v in nx.topological_sort(G):
        if v not in X_v:
            continue
        
        Vup = nx.ancestors(Gsplit, v) | {v}
        
        # Remove the nodes in Vup from Gsplit
        Gsplit = nx.subgraph(Gsplit, [node for node in Gsplit.nodes if node not in Vup])
        
        # Add the length of the upper subgraph to Gup_sizes
        Gup_sizes.append(len(Vup))
    
    if Gsplit.number_of_nodes() != 0:
        Gup_sizes.append(len(Gsplit))
    
    return Gup_sizes
