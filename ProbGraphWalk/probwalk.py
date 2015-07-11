'''
Created on Oct 3, 2012

@author: imaveek
'''

import math
import random

class ProbWalk(object):
    '''
    ProbWalk location multi-agent coverage algorithm
    '''

    def __init__(self, _init_coverageProbMap, sflist, goal_cells, nodeCells, arena, nodesCovered):
        '''
        Constructor
        '''
        # Initialize attributes
        self.c_map = _init_coverageProbMap
        self.arena = arena
        self.arenaMap = arena.gridmap.map
        
        # Initialize est post dictionary
        self.dr_estimated_pos = {}
        for sf in sflist:
            self.dr_estimated_pos[sf.name] = [sf.x, sf.y]
            
        self.last_goal_loc = []
        self.backoff_time_cnt = 0
        self.nodes_covered = nodesCovered

    def command(self, sflist, goalCells, nodeCells):
        '''
        Command as per the coverage algorithm
        '''
        # Give the coverage command
        for sf in sflist:
            # Find current location estimate
            estpos = self.dr_estimated_pos[sf.name]
            loc_cell = self.c_map.xytocell(estpos[0],estpos[1])
            
            # Mark coverage based on real location
            real_loc = self.c_map.xytocell(sf.x, sf.y)
            rId = nodeCells.get(tuple(real_loc), None)
            if (rId != None):
                self.nodes_covered[rId] = True

            # if collided backup randomly jump to a new location
            if sf.hasCollided == True:
                [turn, time, velocity] = self.__computeBackoff(sf)
                sf.setMoveCommand(turn, time, velocity)
                continue
            
            # If old command is complete execute new
            if sf.isMoving == False:
                # Choose neighbors in inverse probability of coverage confidence
                next_loc = self.nextStepToGoal(sf, loc_cell, goalCells, nodeCells, self.last_goal_loc)
                assert next_loc != None, 'Next location is None'
        
                self.last_goal_loc = next_loc
                # Command the sensorfly to setMoveCommand
                [turn, time, velocity] = self.__getTurnTimeVel(sf, estpos, next_loc, self.c_map)
                sf.setMoveCommand(turn, time, velocity)
                

    def update(self, sflist, deltick):
        for sf in sflist:
            pos = self.dr_estimated_pos[sf.name]        
            if sf.hasCollided == False:
                pos[0] = pos[0] + (sf.nowCommand.velocity * deltick) * math.cos(math.radians(sf.pose))
                pos[1] = pos[1] + (sf.nowCommand.velocity * deltick) * math.sin(math.radians(sf.pose))
                if sf.isBackingOff != True:
                    self.backoff_time_cnt = self.backoff_time_cnt - 1 if self.backoff_time_cnt > 1 else 0
            else:
                if sf.isBackingOff == True:
                    self.backoff_time_cnt = 0
                else:
                    self.backoff_time_cnt = self.backoff_time_cnt + 1 if self.backoff_time_cnt < 3 else 3
            self.dr_estimated_pos[sf.name] = pos
#             if self.backoff_time_cnt > 0 : print "Collision Backoff : ", self.backoff_time_cnt


    def __computeBackoff(self, sf):
        '''
        Compute the command to SensorFly after computing the next cell to go on the map
        '''
        # Randomly pick a direction
        newpose = random.choice(list(xrange(0,360,20)))
        turn = newpose - sf.pose
        if (turn < 0):
            turn = turn + 360
        velocity = 1
        time = 2 ** self.backoff_time_cnt
        return [turn, time, velocity]



    def __getTurnTimeVel(self, sf, pos, next_pos, c_map):
        '''
        Compute the command to SensorFly after computing the next cell to go on the map
        '''
        if pos == next_pos:
            return [0, 0, 0]
        delx = int(next_pos[0]) - int(pos[0])
        dely = int(next_pos[1]) - int(pos[1])
        # Get new pose
        newpose = math.degrees(math.atan2(dely,delx))
        turn = newpose - sf.pose
        if (turn < 0):
            turn = turn + 360
        velocity = 1
        time = math.sqrt((delx * c_map.celllength) ** 2 + (dely *
                         c_map.cellbreadth) ** 2) / velocity
        return [turn, time, velocity]
    
    
    def nextStepToGoal(self, sf, loc, goalCells, nodeCells, last_goal_loc):
        '''
        Find the next step to goal as per current location estimate
        '''
        # Find loc in node cells and set goal
        rId = nodeCells.get(tuple(loc), None);        
        if (rId != None):
            goal_door = goalCells[rId]
            goal_loc =  goal_door[len(goal_door)/2]
        else:
            goal_loc = last_goal_loc
#         print "SF ", sf.name, " : ", rId
#         print self.nodes_covered
        self.arena.gridmap.markMap(goal_loc[0], goal_loc[1], self.arena.gridmap.v_goal)          
        return goal_loc
    