# dataserver.py
#
# A server to handle REST API requests from the Grafana dashboard
#
import sys
from random import choice, random

from itertools import count

import numpy as np
import pandas as pd
import json
from flask import Flask, request, abort
from flask_cors import CORS
from shapely import centroid

import utils.utils as utils
from dataingest import dataloader
from network.StormwaterGraph import StormwaterGraph as SWGraph
from utils.psql_connection import PostgreSQLConnection
from utils.CoordinateSystem import CoordinateSystem
from sensors.Sensors import Sensor
from placement.GreedyPlacement import *
from networkanalysis.NetworkAnalysis import NetworkAnalysis



api = Flask(__name__)
CORS(api) # Set CORS headers

ITEM_NUM = count(1)

# A cache for stormwater graphs
class GraphsCache: 
    graphs = {}
    
#A cache for sensors
class SensorsCache:
    sensors = {}

################################################################################
################################################################################
# Map Visualization and Properties
################################################################################

# Handles requests from the Add New SWMM Network Panel
@api.route('/add_new_network', methods=['POST'])
def add_new_network():
    if request.method == 'POST':
        data = request.get_json()
        name = data['new_network_name']
        path = data['new_network_path']

        # Coordinate system to change things into latlong
        crs = CoordinateSystem(source=CoordinateSystem.CRS_2230,
                               target=CoordinateSystem.CRS_4326)

        # Create network
        g = SWGraph(path, network_name=name)
        
        # Cache network graph in global variable
        GraphsCache.graphs[name] = g
        
        # Write geojson files
        fgeojson_subc = utils.get_fgeojson(path, item='subcatchments')
        g.write_subcatchments_geojson(fgeojson_subc)
        fgeojson_conduits = utils.get_fgeojson(path, item='conduits')
        g.write_conduits_geojson(fgeojson_conduits)

        # Add network into database
        with PostgreSQLConnection() as conn:

            cur = conn.cursor()

            # Insert into Networks table
            cur.execute(f'''\
INSERT INTO Networks(network) 
VALUES ('{g.network_name}');
''')

            for sid,subc in g.subc.items():
                p = centroid(subc['polygon'])
                pc_x, pc_y = crs(p.x, p.y)
                cur.execute(f'''\
INSERT INTO Subcatchments(sid, network, outlet, area, centroid_x, centroid_y)
VALUES  ('{sid}', '{g.network_name}', '{subc["data"].connection}', {subc["data"].area}, {pc_x}, {pc_y});
''')

            comp = NetworkAnalysis.branching_complexity(g)
            cent = NetworkAnalysis.betweenness_centrality(g)
            for nid,node in g.nodes().items():
                xy = crs(*node['pos'])
                cur.execute(f'''\
INSERT INTO Junctions(nid, network, elevation, maxdepth, branchingcomplexity, betweennesscentrality, x,y)
VALUES  ('{nid}', '{g.network_name}', {node["data"].invert_elevation}, {node["data"].full_depth},{comp[nid]},{cent[nid]}, {xy[0]},{xy[1]});
''')
    
            for edge in g.edges().values():
                cur.execute(f'''\
INSERT INTO Conduits(eid, network, src, dst, length, roughness)
VALUES  ('{edge["data"].linkid}', '{g.network_name}', '{edge["data"].connections[0]}', '{edge["data"].connections[1]}', 0,0);
''')

            for sid in g.subc:
                for landuse,percent in g.landuse_percent[sid].items():
                    area = g.landuse_area[sid][landuse]
                    cur.execute(f'''\
INSERT INTO Coverages(sid, network, landuse, percent, area)
VALUES  ('{sid}', '{g.network_name}', '{landuse}', {percent}, {area});
''')
                    
            cur.execute(f'''\
INSERT INTO Log(item,description)
VALUES ({next(ITEM_NUM)}, 'Added network "{name}"');

DELETE FROM Current_network;
INSERT INTO Current_network(value) 
VALUES ('{name}');

''')
            

            
        return "success"

        
    # If everything else fails, then just raise a 404 error (not found)
    abort(404)

@api.route('/select_network', methods=['GET', 'POST'])
def select_network():
    if request.method == 'GET':
        with PostgreSQLConnection() as conn:
            cur = conn.cursor()
            cur.execute('''
SELECT network from Networks 
ORDER BY network;''')
            return cur.fetchall()

    if request.method == 'POST':
        data = request.get_json()
        name = data['load_network']
        show_conduits = data['show_network_conduits']
        show_subcatchments = data['show_network_subcatchments']
        show_landuse = data['show_network_landuse']
        show_placement = data['show_network_placement']
        
        # Update the Current database by dropping all rows and reinserting
        with PostgreSQLConnection() as conn:
            cur = conn.cursor()
            cur.execute(f'''\
DELETE FROM Current_network;
INSERT INTO Current_network(value) 
VALUES ('{name}');
            
DELETE FROM Current_show_conduits;
INSERT INTO Current_show_conduits(network, value) 
VALUES ('{name}', '{show_conduits}');

DELETE FROM Current_show_subcatchments;
INSERT INTO Current_show_subcatchments(network, value) 
VALUES ('{name}', '{show_subcatchments}');

DELETE FROM Current_show_landuse;
INSERT INTO Current_show_landuse(network, value) 
VALUES ('{name}', '{show_landuse}');

DELETE FROM Current_show_placement;
INSERT INTO Current_show_placement(network, value) 
VALUES ('{name}', '{show_placement}');
''')
            

            cur.execute(f'''\
INSERT INTO Log(item,description)
VALUES ({next(ITEM_NUM)}, 'Loaded network "{name}"');
''')

            g = GraphsCache.graphs[name]
            comp = NetworkAnalysis.branching_complexity(g)
            cent = NetworkAnalysis.betweenness_centrality(g)
            cur.execute(f'''\
DELETE FROM NetworkMetrics WHERE network = '{name}';
''')

            for nid in g.nodes():
                cur.execute(f'''\
INSERT INTO NetworkMetrics (network, nid, flowcomplexity, betweennesscentrality)
VALUES ('{name}', '{nid}', {comp[nid]}, {cent[nid]});
''')
        
        return "success"
    
    abort(404)

@api.route('/mapviz/<item>', methods=['GET'])
def mapviz(item):
    if request.method == 'GET':
        
        with PostgreSQLConnection() as conn:
            cur = conn.cursor()
            
            # Check if subcatchment should be returned
            cur.execute('SELECT value FROM Current_show_subcatchments;')
            (show,) = cur.fetchone()
            if item == 'subcatchments' and not show:
                return {}
            
            # Check if conduit should be returned
            cur.execute('SELECT value FROM Current_show_conduits;')
            (show,) = cur.fetchone()
            if item == 'conduits' and not show:
                return {}
            
            cur.execute('SELECT value FROM Current_network;')
            (name,) = cur.fetchone()
            
            g = GraphsCache.graphs[name]
            fgeojson = utils.get_fgeojson(g.fswmm, item=item)
            with open(fgeojson) as f:
                return json.load(f)
            
    abort(404)

################################################################################
################################################################################
# Objectives and Constraints
################################################################################

@api.route('/sensors', methods=['POST'])
def sensors():
    if request.method == 'POST':
        data = request.get_json()
        
        with PostgreSQLConnection() as conn:
            cur = conn.cursor()
            
            cur.execute('SELECT value FROM Current_network;')
            (name,) = cur.fetchone()
            
            sname = data['sensor_name']
            stype = data['sensor_type']
            scost = data['sensor_cost']
            ssen  = data['sensor_sensitivity']
            
            SensorsCache.sensors[sname] = Sensor(ssen,stype,1,scost)
            
            cur.execute(f'''\
INSERT INTO Sensors(network,name,type,cost,sensitivity)
VALUES ('{name}', '{sname}', '{stype}', {scost}, {ssen});
''')
            cur.execute(f'''\
INSERT INTO Log(item,description)
VALUES ({next(ITEM_NUM)}, 'Added sensor "{sname}"')
''')
        
        return 'success'

    abort(404)

@api.route('/sensors_csv', methods=['POST'])
def sensors_csv():
    if request.method == 'POST':
        data = request.get_json()
        
        with PostgreSQLConnection() as conn:
            cur = conn.cursor()
            
            cur.execute('SELECT value FROM Current_network;')
            (name,) = cur.fetchone()
            
            fsensors = data['sensors_csv']
            
            for _,r in pd.read_csv(fsensors).iterrows():
                
                sname = r['Sensor Name']
                stype = r['Sensor Type']
                scost = r['Sensor Cost']
                ssen  = r['Sensor Sensitivity']
                
                SensorsCache.sensors[sname] = Sensor(ssen,stype,1,scost)

                cur.execute(f'''\
INSERT INTO Sensors(network,name,type,cost)
VALUES ('{name}', '{sname}', '{stype}', {scost}, {ssen});
''')
            cur.execute(f'''\
INSERT INTO Log(item,description)
VALUES ({next(ITEM_NUM)}, 'Added sensors from "{fsensors}"')
''')
        
        return 'success'

    abort(404)
            
@api.route('/constraints', methods=['POST'])
def constraints():
    if request.method == 'POST':
        data = request.get_json()
        
        with PostgreSQLConnection() as conn:
            cur = conn.cursor()

            cur.execute('SELECT value FROM Current_network;')
            (name,) = cur.fetchone()
            
            if 'budget_constraint' in data and data['budget_constraint']:
                cur.execute(f'''
DELETE FROM Constraints
WHERE description = 'Max Budget';
INSERT INTO Constraints(network,description,value)
VALUES ('{name}', 'Max Budget', '{data["budget_constraint"]}');
''')
                cur.execute(f'''\
INSERT INTO Log(item,description)
VALUES ({next(ITEM_NUM)}, 'Added budget constraint: "{data["budget_constraint"]}"')
''')
            
            if '+location_constraint' in data and data['+location_constraint'] != 'null':
                cur.execute(f'''
INSERT INTO Constraints(network,description,value)
VALUES ('{name}', 'Must add location', '{data["+location_constraint"]}');
''')
                cur.execute(f'''\
INSERT INTO Log(item,description)
VALUES ({next(ITEM_NUM)}, 'Added location constraint(+): "{data["+location_constraint"]}"')
''')

            if '-location_constraint' in data and data['-location_constraint'] != 'null':
                cur.execute(f'''
INSERT INTO Constraints(network,description,value)
VALUES ('{name}', 'Must avoid location', '{data["-location_constraint"]}');
''')
                cur.execute(f'''\
INSERT INTO Log(item,description)
VALUES ({next(ITEM_NUM)}, 'Added location constraint(-): "{data["-location_constraint"]}"')
''')

            if 'reset_location_constraint' in data:
                print("RESET LOC CONSTRAINT NOT YET IMPLEMENTED")
        
        return 'success'
    
    abort(404)

@api.route('/constraints_csv', methods=['POST'])
def constraints_csv():
    if request.method == 'POST':
        data = request.get_json()
        
        with PostgreSQLConnection() as conn:
            cur = conn.cursor()

            for _,r in pd.read_csv(data['constraints_csv']).iterrows():
                if r['Constraint'] == 'Max Budget':
                    cur.execute(f'''
    INSERT INTO Constraints(network,description,value)
    VALUES ('{r["Network"]}', 'Max Budget', '{r["Value"]}');
    ''')
                
                if r['Constraint'] == 'Must add location':
                    cur.execute(f'''
    INSERT INTO Constraints(network,description,value)
    VALUES ('{r["Network"]}', 'Must add location', '{r["Value"]}');
    ''')

                if r['Constraint'] == 'Must avoid location':
                    cur.execute(f'''
    INSERT INTO Constraints(network,description,value)
    VALUES ('{r["Network"]}', 'Must avoid location', '{r["Value"]}');
    ''')

            cur.execute(f'''\
INSERT INTO Log(item,description)
VALUES ({next(ITEM_NUM)}, 'Added constraints from file: "{data["constraints_csv"]}"')
''')

        return 'success'
    
    abort(404)

@api.route('/futurepop', methods=['POST'])
def futurepop():
    if request.method == 'POST':
        data = request.get_json()
        landuse = data['future_pop_landuse_name']
        percent = data['future_pop_landuse_percent']
        
        # Update the Current database by dropping all rows and reinserting
        with PostgreSQLConnection() as conn:
            cur = conn.cursor()

            cur.execute('SELECT value FROM Current_network;')
            (name,) = cur.fetchone()

            cur.execute(f'''\
INSERT INTO FuturePopulations(network,landuse,percent) 
VALUES ('{name}', '{landuse}', {percent})
ON CONFLICT (network,landuse) DO 
    UPDATE SET percent = FuturePopulations.percent;
''')
            
        return 'success'
        
    abort(404)

@api.route('/futurepop_csv', methods=['POST'])
def futurepop_csv():
    if request.method == 'POST':
        data = request.get_json()
        
        with PostgreSQLConnection() as conn:
            cur = conn.cursor()
            
            for _,r in pd.read_csv(data['populations_csv']).iterrows():

                cur.execute('SELECT value FROM Current_network;')
                (name,) = cur.fetchone()
                
                cur.execute(f'''\
INSERT INTO FuturePopulations(network,landuse,percent) 
VALUES ('{name}', '{r["Landuse"]}', {r["Percent"]})
ON CONFLICT (network,landuse) DO 
    UPDATE SET percent = FuturePopulations.percent;
''')
            
        return 'success'
        
    abort(404)

################################################################################
################################################################################
# Placement
################################################################################

@api.route('/placement', methods=['POST'])
def placement():
    if request.method == 'POST':
        data = request.get_json()
        
        if data['objective'] == 'coverage':
            
            with PostgreSQLConnection() as conn:
                cur = conn.cursor()
                
                cur.execute('SELECT value FROM Current_network;')
                (name,) = cur.fetchone()
                
                g = GraphsCache.graphs[name]
                
                sensors = SensorsCache.sensors.values()
                
                cur.execute('''
SELECT Constraints.value 
FROM Constraints 
WHERE Constraints.description = 'Max Budget'
''')
# INNER JOIN Current_network.value = Constraints.network
                (budget,) = cur.fetchone()
                budget = int(budget)
                
                p = GreedyCoveragePlacement.do_placement(g, sensors, budget)
            
                cur.execute(f'''
DELETE FROM Placements WHERE network = '{name}'
''')
                
                for n,s in p.items():
                    cur.execute(f'''\
INSERT INTO Placements(nodeid,network,sensorid)
VALUES ('{n}', '{g.network_name}', '{s}');
''')
                    
                cur.execute(f'''\
DELETE FROM NetworkMetrics_WithPlacement WHERE network = '{name}';
''')

                cov = NetworkAnalysis.coverage(g, placement=p) 
                cov_percentage = {n : len(s) + 1
                                  for n,s in cov.items()}
                comp = NetworkAnalysis.branching_complexity(g, placement=p)
                cent = NetworkAnalysis.betweenness_centrality(g, placement=p)
                for nid in g.nodes():
                    prev_node = np.mean([0] + [comp[x] for x in g.predecessors(nid)])
                    cur.execute(f'''\
INSERT INTO NetworkMetrics_WithPlacement (network, nid, flowcomplexity, coverage, betweennesscentrality)
VALUES ('{name}', '{nid}', {prev_node}, {cov_percentage[nid]}, {cent[nid]});
''')
                    
                agg_comp = np.mean(list(comp.values()))
                agg_cov  = set()
                for x in cov.values():
                    agg_cov |= x
                agg_cov = len(agg_cov) / g.number_of_nodes() * 100
                agg_cent = np.mean(list(cent.values()))
                
                cur.execute(f'''
DELETE FROM Agg_NetworkMetrics_WithPlacement WHERE network = '{name}';
INSERT INTO Agg_NetworkMetrics_WithPlacement (network, flowcomplexity, coverage, betweennesscentrality)
VALUES ('{name}', {agg_comp}, {agg_cov}, {agg_cent});
''')

                cur.execute(f'''\
INSERT INTO Log(item,description)
VALUES ({next(ITEM_NUM)}, 'Generating placement for coverage... ')
''')

        if data['objective'] == 'flowcomplexity':
            
            with PostgreSQLConnection() as conn:
                cur = conn.cursor()
                
                cur.execute('SELECT value FROM Current_network;')
                (name,) = cur.fetchone()
                
                g = GraphsCache.graphs[name]
                
                sensors = SensorsCache.sensors.values()
                
                cur.execute('''
SELECT Constraints.value 
FROM Constraints 
WHERE Constraints.description = 'Max Budget'
''')
# INNER JOIN Current_network.value = Constraints.network
                (budget,) = cur.fetchone()
                budget = int(budget)
                
                p = GreedyBranchingComplexityPlacement.do_placement(g, sensors, budget)
            
                cur.execute(f'''
DELETE FROM Placements WHERE network = '{name}'
''')
                
                for n,s in p.items():
                    cur.execute(f'''\
INSERT INTO Placements(nodeid,network,sensorid)
VALUES ('{n}', '{g.network_name}', '{s}');
''')
                    
                cur.execute(f'''\
DELETE FROM NetworkMetrics_WithPlacement WHERE network = '{name}';
''')

                cov = NetworkAnalysis.coverage(g, placement=p) 
                cov_percentage = {n : len(s) 
                                  for n,s in cov.items()}
                comp = NetworkAnalysis.branching_complexity(g, placement=p)
                cent = NetworkAnalysis.betweenness_centrality(g, placement=p)
                for nid in g.nodes():
                    prev_node = np.mean([0] + [comp[x] for x in g.predecessors(nid)])
                    cur.execute(f'''\
INSERT INTO NetworkMetrics_WithPlacement (network, nid, flowcomplexity, coverage, betweennesscentrality)
VALUES ('{name}', '{nid}', {prev_node}, {cov_percentage[nid]}, {cent[nid]});
''')
                    
                agg_comp = np.mean(list(comp.values()))
                agg_cov  = set()
                for x in cov.values():
                    agg_cov |= x
                agg_cov = len(agg_cov) / g.number_of_nodes() * 100
                agg_cent = np.mean(list(cent.values()))
                
                cur.execute(f'''
DELETE FROM Agg_NetworkMetrics_WithPlacement WHERE network = '{name}';
INSERT INTO Agg_NetworkMetrics_WithPlacement (network, flowcomplexity, coverage, betweennesscentrality)
VALUES ('{name}', {agg_comp}, {agg_cov}, {agg_cent});
''')

                cur.execute(f'''\
INSERT INTO Log(item,description)
VALUES ({next(ITEM_NUM)}, 'Generating placement for flow complexity... ')
''')
                
                
                
                
        return 'success'

    abort(404)

@api.route('/add_sensor', methods=['POST'])
def add_sensor():
    if request.method == 'POST':
        data = request.get_json()
        
        with PostgreSQLConnection() as conn:
            
            cur = conn.cursor()

            cur.execute('SELECT value FROM Current_network;')
            (name,) = cur.fetchone()
            
            cur.execute(f'''
INSERT INTO Placements(network, nodeid, sensorid)
VALUES ('{name}', '{data["add_sensor_node"]}', '{data["add_sensor_type"]}');
''')
            cur.execute(f'''\
INSERT INTO Log(item,description)
VALUES ({next(ITEM_NUM)}, 'Adding sensor "{data["add_sensor_node"]}"')
''')
            
        return 'success'

    abort(404)

@api.route('/remove_sensor', methods=['POST'])
def remove_sensor():
    if request.method == 'POST':
        data = request.get_json()
        
        with PostgreSQLConnection() as conn:
            
            cur = conn.cursor()

            cur.execute('SELECT value FROM Current_network;')
            (name,) = cur.fetchone()
            
            cur.execute(f'''
DELETE FROM Placements
WHERE 
    Placements.network = '{name}' AND 
    Placements.nodeid = '{data["remove_sensor_node"]}' AND
    Placements.sensorid = '{data["remove_sensor_type"]}';
''')
            cur.execute(f'''\
INSERT INTO Log(item,description)
VALUES ({next(ITEM_NUM)}, 'Removing sensor "{data["remove_sensor_node"]}"')
''')
            
        return 'success'

    abort(404)



if __name__ == '__main__':
    
    dataloader.run_setup()
    # dataloader.TEST_add_many_networks(GraphsCache.graphs)

    API_HOST = 'localhost'
    API_PORT = 3001

    api.run(host=API_HOST, port=API_PORT)

