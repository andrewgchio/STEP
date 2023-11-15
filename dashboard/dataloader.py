# main.py

from random import choice
from getpass import getpass

from utils.psql_connection import PostgreSQLConnection

from network.StormwaterGraph import StormwaterGraph as SWGraph
from main.PARAMS import *

from tqdm import tqdm
from utils.CoordinateSystem import CoordinateSystem
from shapely import centroid
from networkanalysis.NetworkAnalysis import NetworkAnalysis

def drop_swmm_tables(cur):
    '''
    Drop all tables in the database
    '''
    cur.execute('''\
DROP TABLE IF EXISTS Placements;
DROP TABLE IF EXISTS Coverages;
DROP TABLE IF EXISTS Conduits;
DROP TABLE IF EXISTS Subcatchments;
DROP TABLE IF EXISTS Junctions;
DROP TABLE IF EXISTS Networks;
DROP TABLE IF EXISTS Log;

DROP TABLE IF EXISTS NetworkMetrics;
DROP TABLE IF EXISTS NetworkMetrics_WithPlacement;
DROP TABLE IF EXISTS Agg_NetworkMetrics_WithPlacement;

DROP TABLE IF EXISTS Sensors;
DROP TABLE IF EXISTS Constraints;
DROP TABLE IF EXISTS FuturePopulations;

DROP TABLE IF EXISTS Current_network;
DROP TABLE IF EXISTS Current_show_conduits;
DROP TABLE IF EXISTS Current_show_subcatchments;
DROP TABLE IF EXISTS Current_show_landuse;
DROP TABLE IF EXISTS Current_show_placement;
''')

def load_swmm_tables(cur):
    '''
    Initialize the tables for the postgres database
    '''
    
    cur.execute('''\
CREATE TABLE Junctions (
    nid       VARCHAR(64),
    network   VARCHAR(64),
    elevation REAL, maxdepth  REAL,
    branchingcomplexity REAL,
    betweennesscentrality REAL,
    x         REAL,
    y         REAL,
    
    PRIMARY KEY (nid, network)
);
''')
    
    # TODO: add polygons? -- to draw
    cur.execute('''\
CREATE TABLE Subcatchments (
    sid      VARCHAR(64),
    network  VARCHAR(64),
    outlet   VARCHAR(64),
    area     REAL,
    centroid_x REAL,
    centroid_y REAL,

    PRIMARY KEY (sid, network)
);
''')
    
    cur.execute('''\
CREATE TABLE Conduits (
    eid VARCHAR(64),
    network  VARCHAR(64),
    src VARCHAR(64),
    dst VARCHAR(64),
    length REAL,
    roughness REAL,
    
    PRIMARY KEY (eid, network)
);
''')
    # CONSTRAINT fk_src
    #     FOREIGN KEY (src)
    #         REFERENCES Junctions(nid)
    #         ON DELETE SET NULL,
    # CONSTRAINT fk_dst
    #     FOREIGN KEY (dst)
    #         REFERENCES Junctions(nid)
    #         ON DELETE SET NULL   

    cur.execute('''\
CREATE TABLE Coverages (
    sid     VARCHAR(64),
    network VARCHAR(64),
    landuse VARCHAR(64),
    percent REAL,
    area    REAL,
    
    CONSTRAINT fk_sid
        FOREIGN KEY (sid, network)
            REFERENCES Subcatchments(sid, network)
            ON DELETE SET NULL
);
''')
    
    cur.execute('''\
CREATE TABLE Placements (
    nodeid   VARCHAR(64),
    network  VARCHAR(64),
    sensorid VARCHAR(64),
    
    
    CONSTRAINT fk_nodeid
        FOREIGN KEY (nodeid, network)
            REFERENCES Junctions(nid, network)
            ON DELETE SET NULL
);
''')
#    PRIMARY KEY (nodeid, network, sensorid),
    
    cur.execute('''\
CREATE TABLE Networks (
    network VARCHAR(64),
    
    PRIMARY KEY (network)
);
''')

    cur.execute('''\
SET TIMEZONE='America/Los_angeles';
CREATE TABLE Log (
    item INT,
    description VARCHAR(128)
);
''')

    cur.execute('''\
CREATE TABLE NetworkMetrics (
    network VARCHAR(64),
    nid VARCHAR(64),
    flowcomplexity REAL,
    betweennesscentrality REAL
);

CREATE TABLE NetworkMetrics_WithPlacement (
    network VARCHAR(64),
    nid VARCHAR(64),
    flowcomplexity REAL,
    coverage REAL,
    betweennesscentrality REAL
);

CREATE TABLE Agg_NetworkMetrics_WithPlacement (
    network VARCHAR(64),
    flowcomplexity REAL,
    coverage REAL,
    betweennesscentrality REAL
);
''')


    cur.execute('''\
CREATE TABLE Sensors (
    network VARCHAR(64),
    name VARCHAR(64),
    type VARCHAR(64),
    cost REAL,
    accuracy REAL,
    sensitivity REAL
);

CREATE TABLE Constraints (
    network VARCHAR(64),
    description VARCHAR(64),
    value VARCHAR(64)
);

CREATE TABLE FuturePopulations (
    network VARCHAR(64),
    landuse VARCHAR(64),
    percent REAL,

    PRIMARY KEY (network, landuse)
);
''')
    
    cur.execute('''\
CREATE TABLE Current_network(network VARCHAR(64), value VARCHAR(64));
CREATE TABLE Current_show_conduits(network VARCHAR(64), value BOOLEAN);
CREATE TABLE Current_show_subcatchments(network VARCHAR(64), value BOOLEAN);
CREATE TABLE Current_show_landuse(network VARCHAR(64), value VARCHAR(64));
CREATE TABLE Current_show_placement(network VARCHAR(64), value BOOLEAN);
''')

# INSERT INTO Current_show_conduits(network, value) VALUES ('t');
# INSERT INTO Current_show_subcatchments(network, value) VALUES ('t');
# INSERT INTO Current_show_placement(network, value) VALUES ('t');

def load_swmm_data(g, cur):

    # Coordinate system to change things into latlong
    crs = CoordinateSystem(source=CoordinateSystem.CRS_2230,
                           target=CoordinateSystem.CRS_4326)
    
    tqdm.write('\nInserting into Networks table:')
    cur.execute(f'''\
INSERT INTO Networks(network)
VALUES ('{g.network_name}');
''')

    tqdm.write('\nInserting into Subcatchments table:')
    for sid,subc in tqdm(g.subc.items()):
        p = centroid(subc['polygon'])
        pc_x, pc_y = crs(p.x, p.y)
        cur.execute(f'''\
INSERT INTO Subcatchments(sid, network, outlet, area, centroid_x, centroid_y)
VALUES  ('{sid}', '{g.network_name}', '{subc["data"].connection}', {subc["data"].area}, {pc_x}, {pc_y});
''')

    tqdm.write('\nInserting into Junctions table:')
    comp = NetworkAnalysis.branching_complexity(g)
    cent = NetworkAnalysis.betweenness_centrality(g)
    for nid,node in tqdm(g.nodes().items()):
        xy = crs(*node['pos'])
        cur.execute(f'''\
INSERT INTO Junctions(nid, network, elevation, maxdepth, branchingcomplexity, betweennesscentrality, x,y)
VALUES  ('{nid}', '{g.network_name}', {node["data"].invert_elevation}, {node["data"].full_depth},{comp[nid]}, {cent[nid]},{xy[0]},{xy[1]});
''')
    
    tqdm.write('\nInserting into Conduits table:')
    for edge in tqdm(g.edges().values()):
        cur.execute(f'''\
INSERT INTO Conduits(eid, network, src, dst, length, roughness)
VALUES  ('{edge["data"].linkid}', '{g.network_name}', '{edge["data"].connections[0]}', '{edge["data"].connections[1]}', 0,0);
''')


    tqdm.write('Inserting into Coverages table:')
    for sid in tqdm(g.subc):
        for landuse,percent in g.landuse_percent[sid].items():
            area = g.landuse_area[sid][landuse]
            cur.execute(f'''\
INSERT INTO Coverages(sid, network, landuse, percent, area)
VALUES  ('{sid}', '{g.network_name}', '{landuse}', {percent}, {area});
''')
            
    # Try out the random placement first
#     tqdm.write('Inserting into Placements table')
#     for i in range(10):
#         nodeid = choice(list(g.nodes))
#         sensorid = 1
#         cur.execute(f'''\
# INSERT INTO Placements(nodeid,network,sensorid)
# VALUES ('{nodeid}', '{g.network_name}', '{sensorid}');
# ''')

def run_setup():
            
    # Connect to the postgres server
    with PostgreSQLConnection() as conn:
        
        # Create cursor
        cur = conn.cursor()
        
        drop_swmm_tables(cur)

        load_swmm_tables(cur)
        
        conn.commit()

def TEST_add_many_networks(graphcache):
    with PostgreSQLConnection() as conn:
        cur = conn.cursor()

        for fswmm,name in zip(FSWMM, FSWMM_NAME):
            if name == 'Anaheim':
                continue
            g = SWGraph(fswmm, network_name=name)
            graphcache[name] = g
            load_swmm_data(g, cur)
        
if __name__ == '__main__':
    
    run_setup()
    
    
    