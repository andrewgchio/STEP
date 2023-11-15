from pyproj import CRS
from pyproj.transformer import Transformer

class CoordinateSystem:
    
    # OpenStreetMaps - latlong
    CRS_4326 = CRS.from_epsg(4326)
    
    # SWMM - NAD83 CA State Plane VI feet
    CRS_2230 = CRS.from_epsg(2230)
    
    # GeoMap - flat latlong
    CRS_3857 = CRS.from_epsg(3857)

    
    def __init__(self, source, target):
        '''
        Initialize the coordinate system 
        '''
        self.source = CRS.from_epsg(source) if type(source) is int else source
        self.target = CRS.from_epsg(target) if type(target) is int else target
    
        self.transform = Transformer.from_crs(self.source, self.target)

    def __call__(self, x, y):
        return self.transform.transform(x,y)
