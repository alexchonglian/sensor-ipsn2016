'''
Created on Sep 27, 2012

@author: imaveek
'''

from Sim.gridmap import GridMap
import DrunkWalk.drunkwalk as dw
import numpy as np
import os
import random



class Controller(object):
    '''
    Simulation Engine
    '''

    def __init__(self, arena):
        '''
        Constructor
        Params: arena - Arena object for the simulation
        '''
        self.arena = arena
        self.explorers = list()
        self.anchors = list()
        self.base = [1, 1]
        self.deltick = []
        

    def addExplorer(self, sf):
        '''
        Add a SensorFly to the simulation
        '''
        self.explorers.append(sf)


    def addExplorers(self, explorers):
        '''
        Add list of SensorFly objects in the simulation
        '''
        for sf in explorers:
            self.explorers.append(sf)


    def removeExplorer(self, name):
        '''
        Remove a SensorFly from the simulation
        '''
        self.explorers = filter(lambda sf: sf.name != name, self.explorers)


    def setExplorerStartLocation(self, base):
        # this function sets start locations of explorers around the given
        # "base" and "radius"
        self.base = base
        sfnum = len(self.explorers)
        radius = sfnum * 2

        # get x range
        if base[0] < radius / 2 + 2:
            x_range = range(1, radius)
        elif base[0] > self.arena.gridmap.gridlength - 2 - radius / 2:
            x_range = range(self.arena.gridmap.gridlength -
                            radius - 1, self.arena.gridmap.gridlength - 2)
        else:
            x_range = range(base[0] - radius / 2, base[0] + radius / 2 - 1)

        # get y range
        if base[1] < radius / 2 + 2:
            yrange = range(1, radius)
        elif base[1] > self.arena.gridmap.gridbreadth - 2 - radius / 2:
            yrange = range(self.arena.gridmap.gridbreadth -
                           radius - 1, self.arena.gridmap.gridbreadth - 2)
        else:
            yrange = range(base[1] - radius / 2, base[1] + radius / 2 - 1)

        for sf in self.explorers:
            i = 0
            while i < 100:
                x = random.sample(set(x_range), 1)[0]
                y = random.sample(set(yrange), 1)[0]
                if self.arena.gridmap.map[x][y] != self.arena.gridmap.v_obstacle:
                    sf.x = x
                    sf.y = y
                    break
                i += 1
            if i >= 100:
                return False
        return True


    def addAnchor(self, sf):
        '''
        Add a SensorFly to the simulation
        '''
        self.anchors.append(sf)


    def addAnchors(self, anchors):
        '''
        Add list of SensorFly objects in the simulation
        '''
        for sf in anchors:
            self.anchors.append(sf)


    def removeAnchor(self, name):
        '''
        Remove a SensorFly from the simulation
        '''
        self.anchors = filter(lambda sf: sf.name != name, self.anchors)
        
        
    def getExplorers(self):
        return self.explorers
    
    
    def getAnchors(self):
        return self.anchors
    

    def run(self, num_ticks, case, pbar, it):
        '''
        Run the simulation
        Params: runtime - time to run in seconds
                deltick - one tick in seconds
        '''

        self.c_map = GridMap(np.zeros(self.arena.gridmap.map.shape))
        self.deltick = case.deltick

        # Change the algorithm here
        algo = dw.DrunkWalk(self.explorers, self.c_map, case)
        # Display
        if case.is_display_on_real:
            self.arena.gridmap.displayInit()
        record = []
        # Main loop
        for tick in range(0, num_ticks):
            pbar.update((num_ticks * it) + tick)
            algo.command()  #get command and set command for each explorer
            self.updateReal()   #execute command and update SensorFly location in real arena
            algo.update()   #Update the estimates as per command 
            # Record data
            if case.goal_graph:
                [pct_covered, is_all_covered] = case.goal_graph.getCoverage()
            else:
                pct_covered = 0
            
#             goal_reached = [sf.is_goal_reached for sf in self.explorers]
            
            sig_match_cnt = sum([sf.sig_match_cnt for sf in self.explorers])
            
            for sf in self.explorers:
                record.append([tick, int(sf.name), sf.xy[0], sf.xy[1], \
                               sf.pf_estimated_xy[0], sf.pf_estimated_xy[1], \
                               sf.dr_estimated_xy[0], sf.dr_estimated_xy[1], pct_covered, sig_match_cnt])
            # Visualize
            if case.is_display_on_real:
                self.arena.gridmap.displayUpdate()
            if case.stop_on_all_covered and case.goal_graph and is_all_covered:
                break
        return record


    def cls(self):
        os.system(['clear', 'cls'][os.name == 'nt'])
        # now, to clear the screen

    def updateReal(self):
        '''
        Execute command and update SensorFly location in real arena
        '''
        for sf in self.explorers:
            # If SensorFly is dead continue
            if not sf.is_alive or sf.is_goal_reached: 
                continue
            pos = self.arena.gridmap.xytocell(sf.xy)
            self.arena.gridmap.markMap(pos, self.arena.gridmap.v_covered)
            oldpos = pos
            # Update
            sf.update(self.deltick, self.arena)
            # Mark new position in arena
            pos = self.arena.gridmap.xytocell(sf.xy)
            self.coverAllInBetween(pos, oldpos, self.arena.gridmap)
            self.arena.gridmap.markMap(pos, self.arena.gridmap.v_node)


    def coverAllInBetween(self, loc, lastloc, lmap):
        if loc == lastloc:
            return lastloc
        [x, y] = loc
        [x2, y2] = lastloc
        cov_cells = self.bresenhamLine(x, y, x2, y2)
        for c in cov_cells:
            lmap.markMap(c, self.arena.gridmap.v_covered)


    def bresenhamLine(self, x, y, x2, y2):
        # Brensenham line algorithm
        steep = 0
        coords = []
        dx = abs(x2 - x)
        if (x2 - x) > 0:
            sx = 1
        else:
            sx = -1
        dy = abs(y2 - y)
        if (y2 - y) > 0:
            sy = 1
        else:
            sy = -1
        if dy > dx:
            steep = 1
            x, y = y, x
            dx, dy = dy, dx
            sx, sy = sy, sx
        d = (2 * dy) - dx
        for _ in range(0, dx):
            if steep:
                coords.append((y, x))
            else:
                coords.append((x, y))
            while d >= 0:
                y = y + sy
                d = d - (2 * dx)
            x = x + sx
            d = d + (2 * dy)
        coords.append((x2, y2))
        return coords