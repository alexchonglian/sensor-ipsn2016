'''
Created on Oct 4, 2012

@author: imaveek
'''

import math
import numpy as np

class SMController(object):
    '''
    The command logic for multi-robot coverage
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.covStacks = {}
        self.sfPreviousNextLoc = {}
        self.sfPreviousLastLoc = {}
        self.lastNextLoc = {}
        self.thrd_gridIsCoveraged = 2
        self.debugging = False
        
    def useNextLocAndGo(self, sf, loc, nextloc, coverageProbMap, ifBackingOff):
        sf.backingOff = ifBackingOff
        [turn, time, velocity] = self.__getTurnTimeVel(sf, loc, nextloc, coverageProbMap)
        sf.setMoveCommand(turn, time, velocity)
        self.lastNextLoc[sf.name] = nextloc

    def commandExploration(self, sf, sflist, particles, coverageProbMap, nextlocOther):
        # 1) get stack first
        sfstack = []
        if sf.name not in self.covStacks:
            self.covStacks[sf.name] = sfstack
        else:
            sfstack = self.covStacks[sf.name]
            
        # 2) get current loc
        plist = particles[sf.name][0] # get particle lists of the current sensorfly
        loc  = self.locEstimate(plist) # determine sensorfly's location using particles' locations
#        if loc == [2,5]:
#            pass
#        if self.debugging:
#        print "sf loc:" + str(sf.x) + " " + str(sf.y) + "; est: " + str(loc[0]) + " " + str(loc[1])
        # 3) determine if we have collided
        if sf.hasCollided == True: # we have collided, we back off
            coverageProbMap.markMap(loc[0],loc[1],coverageProbMap.vObstacle)
            if self.debugging:
                print "hasCollided"
            # Note: if it has collided, it means we are at the same location as the last iteration,
            # indicating the last append into the stack isn't valid, we should pop out the last element from the stack
            if sf.state != "migration" and len(sfstack) > 1:
                sfstack.pop() # Since the migration mode doesn't have stack, we only do popping in the exploration mode
                sf.tryingToProceed = True # this indicator is used in consecutive iterations to retain locToAvoid
                # 3.1) get the next loc we computed last time
                lastNextLoc = self.lastNextLoc[sf.name]
                sf.locToAvoid.append(lastNextLoc) # Since our current location made us collided, we should mark current location is an obstacle, avoiding us from moving to here once again
                lastloc = sfstack[-1] # when we collide, we step on the same loc as last iteration, so we find the loc before coming to here by using the stack
            else: # migration mode
#                print "===================================================================================== nowhere to go; starting migration"
                [turn, time, velocity, nextloc] = self.computeCommandMigration(sflist, particles, loc, coverageProbMap)
                sf.setMoveCommand(turn, time, velocity)
                nextlocOther.append(nextloc)
                return
        # 4) we haven't collided
        else: # we haven't collided, then we determine what's going to do next
            if self.debugging:
                print "not collided"
            if sf.tryingToProceed == True:# if we haven't collided but tryingToProceed is true, this means we have successfully found a valid cell to jump to
                sf.tryingToProceed = False # clear indicator
                sf.locToAvoid[:] = [] # clean up locToAvoid list
            lastloc = self.lastLocEstimate(plist)
            lastloc = self.coverAllInBetween(loc, lastloc, coverageProbMap, sfstack) # in case we skipped any cells in between, we put them into stack

        # 3.1) we get neighbors excluding the lastloc
        N = self.neighbors(sf, loc, lastloc, coverageProbMap) # get number of neighbors excluding the last loc as well as outsiders
        # 3.2) we determine proper neighbor cell to go to
        n = self.chooseNextStep(N, coverageProbMap, nextlocOther, sf.locToAvoid, "thrd")
        # 3.3) based on what we have chosen as the next cell, we could either go to the next cell or we back off
        if n == None: # we have nowhere to go, we try to back off
            if self.debugging:
                print "next step is none"
            if len(sfstack) > 0:
                if self.debugging:
                    print "backing off: " + str(sfstack)
                nextloc = sfstack.pop()
                self.useNextLocAndGo(sf, loc, nextloc, coverageProbMap, True)
                nextlocOther.append(nextloc)
            else: # our stack is empty, we have nowhere to go, so we enter the migration mode
#                    sf.state = "migration"
#                print "===================================================================================== nowhere to go; starting migration"
                sf.backingOff = False
                [turn, time, velocity, nextloc] = self.computeCommandMigration(sflist, particles, loc, coverageProbMap)
                sf.setMoveCommand(turn, time, velocity)
                nextlocOther.append(nextloc)
        else: # we have chosen a cell to go to
            if self.debugging:
                print "has a new cell to go to"
            sfstack.append(loc)
            nextloc = n
            self.useNextLocAndGo(sf, loc, nextloc, coverageProbMap, False)
            nextlocOther.append(nextloc)

        
    def computeCommandMigration(self, sflist, particles, loc, coverageProbMap):
        sfInExploration = []
        for sf in sflist:
            if sf.state == "exploration":   
                sfInExploration.append(sf)
            
        disMax = -1.0
        if len(sfInExploration) == 0:
            return [0, 0, 0, loc]
        # find the farthest exploring sensorfly
        for sf in sfInExploration:
            plist = particles[sf.name][0] # get particle lists of the current sensorfly
            loc2 = self.locEstimate(plist) # determine sensorfly's location using particles' locations  
            dis =  math.sqrt((loc2[0]-loc[0])**2+(loc2[1]-loc2[1])**2)
            if dis > disMax:
                disMax = dis
                sfMax = sf
        nextloc = self.locEstimate(particles[sfMax.name][0]) # find location of sfMax
        [turn, time, velocity] = self.computeCommandAnyTurnValue(sf, loc, nextloc, coverageProbMap)
        return [turn, time, velocity, nextloc]
                
    def command(self, sflist, particles, coverageProbMap):
        '''
        Command multiple SensorFly's according to probabilistic OMRTC logic
        '''
        nextlocOther = []
        # Recursive step
        for sf in sflist:
#            print sf.state
            if sf.state == "exploration":
                self.commandExploration(sf, sflist, particles, coverageProbMap, nextlocOther)
            elif sf.state == "migration": 
                self.commandMigration(sf, sflist, particles, coverageProbMap, nextlocOther)
        pass
                    
    def chooseNextStep(self, N, coverageProbMap, nextlocOther, locToAvoid, method):
        # choose the neighboring grid that has the lowest coverage prob as the next step
        # nextlocOther represents all "nextloc" of other sensorflies that are at the same location as the current sensorfly;
        # the next step of the current sensorfly shouldn't be the same as that of the other sensorflies if they are at the same location
        if method == "min":
            nMin = None
            probMin = 100000
            for n in N:
                if coverageProbMap.map[n[0]][n[1]] < probMin and n not in nextlocOther and n not in locToAvoid:
                    probMin = coverageProbMap.map[n[0]][n[1]]
                    nMin = n
            return nMin
        elif method == "thrd":
            thrd = 0.8
            for n in N:
                if coverageProbMap.map[n[0]][n[1]] < thrd and n not in nextlocOther and n not in locToAvoid:
                    return n
                                
    def coverAllInBetween(self, loc, lastloc, globalMap, sfstack = None):
        
        if loc == lastloc:
            return lastloc
        
        if lastloc[0] > loc[0]:
            x_range = range(lastloc[0]-1, loc[0], -1)
        else:
            x_range = range(lastloc[0]+1, loc[0], 1)
        
        if lastloc[1] > loc[1]:
            yrange = range(lastloc[1]-1, loc[1], -1)
        else:
            yrange = range(lastloc[1]+1, loc[1], 1)
        
        if len(x_range) == 0 and len(yrange) == 0:
            return lastloc
        
        for xi in x_range:
            pos = [xi, lastloc[1]]
            if sfstack != None:
                sfstack.append(pos)
        for yi in yrange:
            pos = [loc[0], yi]
            if sfstack != None:
                sfstack.append(pos)
        
        if len(x_range) == 0:
            return [loc[0], yrange[-1]]
        elif len(yrange) == 0:
            return [x_range[-1], loc[1]]
        else:
            return [x_range[-1], yrange[-1]]
                
    def __getTurnTimeVel(self, sf, pos, newloc, thisMap):
        '''
        Compute the command to SensorFly after computing the next cell to go on the map
        '''
        oldpos = pos
        newpos = newloc
        if oldpos == newpos:
            return [0, 0, 0]
#        print newpos, oldpos
        delx = newpos[0] - oldpos[0]
        dely = newpos[1] - oldpos[1]
        
        # Get new dir    
        newpose = self.getPose([delx,dely])
        turn = newpose - sf.dir
        if (turn < 0):
            turn = turn + 360
            
        velocity = 1
        time = math.sqrt((delx * thisMap.celllength)**2 + (dely * thisMap.cellbreadth)**2) / velocity
        return [turn, time, velocity]
        
    def getPoseAnyTurnValue(self,dxy):
        if self.debugging:
            print dxy
        dx = dxy[0] * 1.0
        dy = dxy[1] * 1.0
        return math.atan2(dy, dx) / math.pi * 180
        
    def computeCommandAnyTurnValue(self, sf, loc, newloc, thisMap):
        '''
        Compute the command to SensorFly after computing the next cell to go on the map
        '''
        oldpos = loc
        newpos = newloc
        if oldpos == newpos:
            return [0, 0, 0]
#        print newpos, oldpos
        delx = newpos[0] - oldpos[0]
        dely = newpos[1] - oldpos[1]
        
        # Get new dir    
        newpose = self.getPoseAnyTurnValue([delx,dely])
        turn = newpose - sf.dir
        if (turn < 0):
            turn = turn + 360
            
        velocity = 1
        time = math.sqrt((delx * thisMap.gridlength)**2 + (dely * thisMap.gridbreadth)**2) / velocity
        return [turn, time, velocity]
                
    def neighbors(self, sf, loc, lastloc, globalMap):
        '''
        Find neighboring cells in clockwise order from the last position
        '''
        [x,y] = loc
        N = [[x,y+1],[x+1,y],[x,y-1],[x-1,y]]
        perm = sf.perm
        N = [N[perm[0]],N[perm[1]], N[perm[2]], N[perm[3]]]
        N = [n for n in N if n != lastloc and self.outOfArena(n, globalMap) == False]
        return N
        
    def outOfArena(self, N, thisMap): # this checks if the sensorfly's virtual location is out of the virtual map, which shouldn't happen
        X, Y = N[0], N[1]
        if X >= thisMap.gridlength-1 or Y >= thisMap.gridbreadth-1 \
            or X <= 0 or Y <= 0 or thisMap.map[X][Y] == 1:
            return True
        else:
            return False
        
    def locEstimate(self, plist):
        '''
        Estimate location of SensorFly from centroid of all particle positions
        '''
        xlist = []
        ylist = []
        for p in plist:
            xlist.append(p.pos[0])
            ylist.append(p.pos[1])
        x = int(round(np.median(xlist)))
        y = int(round(np.median(ylist)))
        return [x,y]
        

    def lastLocEstimate(self, plist):
        '''
        Estimate last location of SensorFly from centroid of all particle positions
        '''
        xlist = []
        ylist = []
        for p in plist:
            xlist.append(p.lastpos[0])
            ylist.append(p.lastpos[1])
        x = int(round(np.median(xlist)))
        y = int(round(np.median(ylist)))
        return [x,y]

    
    def getPose(self,dxy):
        '''
        Compute dir from the next cell movement
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
