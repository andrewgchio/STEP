from collections import defaultdict

import numpy as np
import pandas as pd

from hymo import hymo

import utils.utils as utils
from main.PARAMS import *
from utils.CoordinateSystem import CoordinateSystem

def read_all_tags():
    taglist = []
    for fswmm, fswmmname in zip(FSWMM, FSWMM_NAME):
        print(f'Loaded Network: {fswmmname}')

        fhymo = hymo.SWMMInpFile(fswmm)
        
        tags = fhymo.tags.reset_index()
        tags['Network'] = fswmmname
        taglist.append(tags)
    
    tags = pd.concat(taglist).reset_index()
    return tags

def make_tagmap(Xtags):
    tagmap = defaultdict(lambda : (None,None))
    for _,r in Xtags.iterrows():
        if r['Object'] == 'Node':
            tagmap[r['Type']] = r['Network'], r['Name']
    return tagmap

def read_all_coords():
    coords = {}
    for fswmm,fswmmname in zip(FSWMM, FSWMM_NAME):
        fhymo = hymo.SWMMInpFile(fswmm)
        for _,r in fhymo.coordinates.reset_index().iterrows():
            coords[r['X_Coord'],r['Y_Coord']] = r['Node'],fswmmname
    return coords
        
def find_nearest(coords, xy):
    return min((utils.dist2(xy, (x,y)), v, netname) 
        for (x,y),(v,netname) in coords.items())
 
if __name__ == '__main__':
    
    utils.setup_pandas()

    Xtags = read_all_tags()
    tagmap = make_tagmap(Xtags)
    coords_map = read_all_coords()
    
    # Only keep relevant node tags
    for fhist in HISTORICAL_DATA:
        
        X = pd.read_excel(fhist, skiprows=1)

        # # Relevant thresholds
        # tolerances_row = X.iloc[1]
        # thresholds = {
        #     'turbidity' : tolerances_row['Turbidity'],
        #     'ec' : tolerances_row['Electrical Conductivity'],
        #     'temperature' : tolerances_row['Water Temperature'],
        #     'velocity' : velocity_simulation,
        #     'depth' : depth_simulation 
        # }
        
        # Drop other irrelevant rows / rename others
        X.drop([0,1,2], inplace=True)
        X.drop(['Unnamed: 0', 'Unnamed: 4', 'Unnamed: 43'], axis=1, inplace=True)
        X.rename({'Unnamed: 1' : 'Site',
                  'Unnamed: 2' : 'Date',
                  'GIS Coordinates' : 'lat',
                  'Unnamed: 6' : 'lon'}, axis=1, inplace=True)

        X['Date'] = X['Date'].apply(lambda x : x.date())

        # Remove NaN rows
        X.dropna(inplace=True)
        
        X = X.reset_index()
        
        # Find out which network they belong to
        X['Network'] = X['Site'].apply(lambda x : tagmap[x][0])
        X['Node'] = X['Site'].apply(lambda x : tagmap[x][1])
        
        print('Unknown sites:', set(X[X['Node'].isna()]['Site']))

        # Use latlon to replace anything else
        crs = CoordinateSystem(source=CoordinateSystem.CRS_4326, 
                               target=CoordinateSystem.CRS_2230)
        X['latlon2'] = X.apply(lambda x : crs(x['lat'], x['lon']), axis=1)
        X['Nearest'] = X['latlon2'].apply(lambda x : find_nearest(coords_map, x))
        X['NearestNode'] = X['Nearest'].apply(lambda x : x[1])
        X['NearestNetwork'] = X['Nearest'].apply(lambda x : x[2])

        # print(X[['Site', 'Network', 'Node', 'NearestNetwork', 'NearestNode']].drop_duplicates())

        X['Network'].fillna(X['NearestNetwork'], inplace=True)
        X['Node'].fillna(X['NearestNode'], inplace=True)

        X = X[['Site', 'Network', 'Node', 'Date', 'Time',
               'Turbidity', 'Electrical Conductivity', 'Water Temperature', 'Discharge Rate']]
        
        print(X.head())

        fhist_save = utils.get_fhistsave(fhist)
        X.to_csv(fhist_save)
        
        print('Done')
        
        