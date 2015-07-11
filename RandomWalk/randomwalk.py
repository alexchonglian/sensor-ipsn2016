'''
Created on 2012-10-4

@author: Administrator
'''
import random

class DrunkWalk(object):
    '''
    Randomwalk coverage algorithm
    '''

    def __init__(self, sflist, cMap, nodesCovered):
        '''
        Constructor
        '''
        self.cMap = cMap
        self.nodes_covered = nodesCovered
           
           
    def command(self, sflist, goalCells, nodeCells):
        '''
        Command as per the coverage algorithm
        '''
        # Give the coverage command
        
        for sf in sflist:
            # Check if SensorFly has reached goal
            # Mark coverage based on real location
            real_loc = self.cMap.xytocell(sf.x, sf.y)
            rId = nodeCells.get(tuple(real_loc), None)
            if (rId != None):
                self.nodes_covered[rId] = True
            
            # deal with wall collision first
            if sf.isMoving == False:
                [turn, time, velocity] = self.getCommandRandom(sf)
                sf.setMoveCommand(turn, time, velocity)
        
        
    def getCommandRandom(self, sf):
        '''
        Compute the command to SensorFly after computing the next cell to go on the map
        '''
        # Randomly pick a direction
        newpose = random.randrange(0,360,20)
        turn = newpose - sf.pose
        if (turn < 0):
            turn = turn + 360
        velocity = 1
        time = random.randint(1,5)
        return [turn, time, velocity]