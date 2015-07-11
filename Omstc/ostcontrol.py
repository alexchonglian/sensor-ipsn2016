'''
Created on 2012-10-5

@author: Administrator
'''
import math
from random import choice

class OSTControl(object):
    '''
    classdocs
    '''
    def __init__(self, sflist):
        '''
        Constructor
        '''
        self.covStacks = {}
            
    def command(self, sflist, cMap):
        '''
        Command multiple SensorFly's according to spreadout algo 
        Idea: 
        '''
        for sf in sflist:
            sfstack = []
            if sf.name not in self.covStacks:
                self.covStacks[sf.name] = sfstack
            else:
                sfstack = self.covStacks[sf.name]
            
            # Get current location
            loc = cMap.xytocell(sf.x, sf.y)
            
            # Get neighbors excluding the lastloc
            nbors, nvals = self.neighbors(sf, loc, cMap)
            
            nextLoc = self.span_tree(nbors, nvals, sfstack)
            assert nextLoc != None, 'Next location is None'
            
            # Command the sensorfly to setMoveCommand
            [turn, time, velocity] = self.__getTurnTimeVel(sf, loc, nextLoc, cMap)
            sf.setMoveCommand(turn, time, velocity) 
            
    def neighbors(self, sf, loc, cMap):
        '''
        Find neighboring cells in clockwise order from the last position
        '''
        [x,y] = loc
        nbors = [[x,y+1],[x+1,y],[x,y-1],[x-1,y]]
        nbors = [n for n in nbors if self.outOfArena(n, cMap) == False]
        nvals = [cMap.map[n[0]][n[1]] for n in nbors]
        return nbors, nvals
    
    def span_tree(self, nbors, nvals, sfstack):
        # If all neighboring cells are explored backtrack
        for c,v in zip(nbors, nvals):
            if v == 0:
                return c
        if sfstack:
            c = sfstack.pop()
        else:
            c = choice(nbors)
        sfstack.append(c) # Store last location
        return c
    
    def outOfArena(self, N, cMap): # this checks if the sensorfly's virtual location is out of the virtual map, which shouldn't happen
        X, Y = N[0], N[1]
        if X >= cMap.gridlength-1 or Y >= cMap.gridbreadth-1 \
            or X <= 0 or Y <= 0:
            return True
        else:
            return False
        
    def __getTurnTimeVel(self, sf, pos, next_pos, c_map):
        '''
        Compute the command to SensorFly after computing the next cell to go on the map
        '''
        if pos == next_pos:
            return [0, 0, 0]
        delx = next_pos[0] - pos[0]
        dely = next_pos[1] - pos[1]
        # Get new pose    
        newpose = self.getPose([delx,dely])
        turn = newpose - sf.pose
        if (turn < 0):
            turn = turn + 360
        velocity = 1
        time = math.sqrt((delx * c_map.celllength)**2 + (dely * c_map.cellbreadth)**2) / velocity
        return [turn, time, velocity]
    
    def getPose(self,dxy):
        '''
        Compute pose from the next cell movement
        '''
        if (dxy == [1,0]):
            return 0
        if (dxy == [0,1]):
            return 90
        if (dxy == [-1,0]):
            return 180
        if (dxy == [0,-1]):
            return 270
        return 0