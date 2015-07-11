'''
Created on Oct 10, 2013

@author: imaveek
'''
import numpy as np

class Magnetometer(object):
    '''
    Magnetometer
    '''


    def __init__(self, sf):
        '''
        Constructor
        '''
        self.sf = sf
        self.noise = sf.noise_mag
        
    
    def getDirection(self):
        '''
        Get the direction with added normal noise
        '''
        return np.random.normal(self.sf.dir, self.noise) % 360
        
        