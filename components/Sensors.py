from random import normalvariate

import pandas as pd

import utils.utils as utils

class Sensor:

    def __init__(self, sid, error, unit, phenomenon, cost):
        '''
        Initialize a sensor with a given: 
            sid: id of the sensor
            error: observations made with some error < error tol
            phenomena: the type of phenomena observed by the sensor
            period: the number of minutes between observations
            cost: the cost of the sensor
        '''
        self.sid = sid
        self.error = error
        self.unit = unit
        self.phenomenon = phenomenon
        self.cost = cost
    
    def __str__(self):
        return f'''\
Sensor {self.sid} (
    error={self.error}{"%" if self.unit == "percent" else self.unit},
    phenomenon={self.phenomenon},
    cost={self.cost}
)'''
    
    def do_observe(self, obs):
        '''
        Produce an observation with the specified error 
        '''
        if self.unit == 'percent': # percentage error
            return max(0.0, obs + normalvariate(0.0, self.error/100 * obs)) 
        else: # absolute error
            return max(0.0, obs + normalvariate(0.0, self.error))
        
class Sensors:
    
    SENSORS = None

    def __init__(self, fsensors=None):

        Sensors.SENSORS = self

        self.data = {}
        
        self.sids = []
        
        if fsensors is not None:
            self.load_sensors(fsensors)
    
    def load_sensors(self, fname):
        Xsensors = pd.read_csv(fname)
        for _,r in Xsensors.iterrows():
            self.sids.append(r['Sensor ID'])
            self.data[r['Sensor ID']] = Sensor(r['Sensor ID'],
                                    r['Accuracy'],
                                    r['Unit'],
                                    r['Phenomenon'],
                                    r['Cost'])
        
    def __str__(self):
        return '\n'.join(map(str, self.data))
    
    def __iter__(self):
        return iter(self.data.items())

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, k):
        return self.data[k]

    def get(self, p):
        '''
        Return the set of sensors that measure phenomenon p
        '''
        return {i:x for i,x in self.data.items() if x.phenomenon == p}

    def get_sids(self):
        return self.sids

@utils.Memoize
def getSensorsMeasuring(p):
    return list(Sensors.SENSORS.get(p).keys())
    