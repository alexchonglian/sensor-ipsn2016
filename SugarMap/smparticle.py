'''
Created on Oct 3, 2012

@author: imaveek
'''
import math
from Sim.sensorfly import SensorFly
from Sim.gridmap import GridMap
class Particle(SensorFly):
    '''
    Particle for sensorfly
    '''

    def __init__(self, sf, init_weight, sensorflyOwnMap, sf0):
        '''
        Constructor
        '''
        self.weight = init_weight
        self.landmarkDict = {}
        # Local coverage map for each particle
        self.localMap = GridMap(sensorflyOwnMap.length, sensorflyOwnMap.breadth, 
                               sensorflyOwnMap.celllength, sensorflyOwnMap.cellbreadth)
        if 1:
            self.pos = self.localMap.xytocellNoBoundaryCheck(sf.x-sf0.x, sf.y-sf0.y)
            self.pos = [self.pos[0] + int(self.localMap.gridlength/2), self.pos[1] + int(self.localMap.gridbreadth/2)]
            self.lastpos = self.localMap.xytocellNoBoundaryCheck(sf.x-sf0.x, sf.y-sf0.y)
            self.lastpos = [self.lastpos[0] + int(self.localMap.gridlength/2), self.lastpos[1] + int(self.localMap.gridbreadth/2)]
        else:
            self.pos = self.localMap.xytocellNoBoundaryCheck(sf.x, sf.y)
            self.lastpos = self.localMap.xytocellNoBoundaryCheck(sf.x, sf.y)
        # SensorFly constructor
        SensorFly.__init__(self, sf.name, sf.x, sf.y, sf.dir, sf.battery, sf.noiseVelocity, sf.noise_turn, \
                           sf.noiseRadio)        
        
        
    def setMoveCommand(self, turn, velocity):
        '''
        Params: turn - the angle to turn in degrees
                time - the time to setMoveCommand in seconds
        '''
        # Set the command to be executed now
        self.nowCommand.turn = self.odometer.turn(turn)
        self.nowCommand.velocity = self.odometer.velocity(velocity)
        self.dir = (self.dir + self.nowCommand.turn) % 360
        
    
    def update(self, sf, deltick):
        '''
        Params: deltick - the time tick in seconds
        '''
        # Move
        if sf.hasCollided == False:
            oldposxy = [self.pos[0], self.pos[1]]
            self.pos[0] = self.pos[0] + (self.nowCommand.velocity * deltick) * math.cos(math.radians(self.dir))
            self.pos[1] = self.pos[1] + (self.nowCommand.velocity * deltick) * math.sin(math.radians(self.dir))
            newposxy = self.pos
            self.updateLocalMap(oldposxy, newposxy)
        # Check if SensorFly has collided
        # TODO: Fix the collision coordinates 
        self.hasCollided = sf.hasCollided
    
    
    def updateLocalMap(self, oldposxy, newposxy):
        '''
        Update covered local map
        '''
        # Mark new position as covered
        # TODO: Add coverage to all cells along path
        opos = self.localMap.xytocell(oldposxy[0],oldposxy[1])
        npos = self.localMap.xytocell(newposxy[0],newposxy[1])
        self.pos = npos
        self.lastpos = opos
        self.localMap.markMap(npos[0],npos[1],1)
        
        