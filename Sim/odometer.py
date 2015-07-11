'''
Created on Oct 3, 2012

@author: imaveek
'''
import numpy as np

class Odometer(object):
    '''
    Odometer sensor
    '''


    def __init__(self, noiseV_, noiseT_):
        '''
        Constructor
        '''
        self.noise_v = noiseV_
        self.noise_t = noiseT_
            
    def velocity(self, velocity):
        '''
        Define odometer sensor model
        '''
        nV = np.abs(velocity * self.noise_v)
        return np.random.normal(velocity, nV)
    
    def turn(self, turn):
        '''
        Define gyro sensor model
        '''
        nT = np.abs(turn * self.noise_t)
        if nT == 0: nT = 0.1
        return np.random.normal(turn, nT)