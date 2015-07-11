'''
Created on Sep 27, 2012

@author: imaveek
'''
from Sim.gridmap import GridMap as gm

class Arena(object):
    '''
    Arena for the simulator
    '''
    def __init__(self, map_array):
        '''
        Params: Length - in number of cells
                Breadth - in number of cells
                Cellsize - area in sq. meters
        '''
        self.gridmap = gm(map_array)
        
        
        
