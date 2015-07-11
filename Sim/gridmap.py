'''
Created on 2012-10-5

@author: Administrator
'''
import numpy as np
import pygame

# Define some colors
black    = (   0,   0,   0)
white    = ( 255, 255, 255)
green    = (   0, 255,   0)
red      = ( 255,   0,   0)
purple   = ( 72, 61, 139)

class GridMap(object):
    '''
    classdocs
    '''
    
    def __init__(self, map_array):
        '''
        Constructor
        ''' 
        self.length = map_array.shape[0]
        self.breadth = map_array.shape[1]
        self.celllength = 1
        self.cellbreadth = 1
        self.gridlength = int(self.length / self.celllength)
        self.gridbreadth = int(self.breadth / self.cellbreadth)
        
        self.width = 800/self.gridbreadth
        self.height = 800/self.gridlength
        self.margin = 0
        
        self.v_covered = 50
        self.v_obstacle = 90
        self.v_node = 150
        self.v_goal = 237
        
        self.map = map_array * self.v_obstacle
        self.num_empty_cells = self.findNumEmptyCells()
        
        self.num_covered = 0
        self.num_empty_cells = self.findNumEmptyCells()
        
        self.change_list = []
        
        # Initialize pygame
        pygame.init()
        
    def __exit__(self, typ, value, traceback):
        pygame.quit ()
      
        
    def markMap(self, xy,val):
        lastval = self.map[xy[0]][xy[1]]
        if (val == self.v_covered and lastval != self.v_covered):
            self.num_covered += 1 # Increment covered cells
        elif (val != self.v_covered and lastval == self.v_covered):
            self.num_covered -= 1 # Decrement covered cells
        self.map[xy[0]][xy[1]] = val
        self.change_list.append([xy[0],xy[1]])
        
        
    def xytocell(self, xy):
        X = int(round(xy[0] / self.celllength))
        Y = int(round(xy[1] / self.cellbreadth))
        if X < 0:
            X = 0
        elif X >= self.gridlength:
            X = self.gridlength-1
        if Y < 0:
            Y=0
        elif Y >= self.gridbreadth:
            Y=self.gridbreadth-1
        return [X,Y]
    
    def xytocellArray(self, xy):
        cell_xy = np.copy(xy)
        cx = cell_xy[:,0]
        cy = cell_xy[:,1]
        # Convert
        cx = np.round(cx / self.celllength)
        cy = np.round(cy / self.cellbreadth)
        # Bound
        cx[cx < 0] = 0
        cx[cx >= self.gridlength] = self.gridlength - 1 
        cy[cy < 0] = 0
        cy[cy >= self.gridbreadth] = self.gridbreadth - 1
        return cell_xy.astype(int)

    
    def xytocellNoBoundaryCheck(self, xy):
        X = int(round(xy[0] / self.celllength))
        Y = int(round(xy[1] / self.cellbreadth))
        return [X,Y]


    def cell2xy(self,XY):
        x = (XY[0] - 1) * self.celllength
        y = (XY[1] - 1) * self.cellbreadth
        return [x,y]
    
    def cell2xyArray(self,cell_xy):
        xy = np.copy(cell_xy)
        x = xy[:,0]
        y = xy[:,1]
        x = (x - 1) * self.celllength
        y = (y - 1) * self.cellbreadth
        return xy.astype(np.float32)
                
    
    def findNumEmptyCells(self):
        numEmpty = np.size(self.map) - np.count_nonzero(self.map)
        return numEmpty
    
    
    def displayInit(self):
        # Set the height and width of the screen
        size=[800,800]
        self.screen=pygame.display.set_mode(size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock=pygame.time.Clock()
        # Set the screen background
        self.screen.fill(black)
        # Draw the grid
        for row in range(self.gridlength):
            for column in range(self.gridbreadth):
                color = white
                if self.map[row][column] == self.v_goal:
                    color = red
                if self.map[row][column] == self.v_node:
                    color = purple
                elif self.map[row][column] == self.v_covered:
                    color = green
                elif self.map[row][column] == self.v_obstacle:
                    color = black
                elif self.map[row][column] > 0 and self.map[row][column] <= 1.1:
                    color = (0,0,255 - int(self.map[row][column] * 255))
                pygame.draw.rect(self.screen,color,\
                                [(self.margin+self.width)*column+self.margin,\
                                 (self.margin+self.height)*row+self.margin,self.width,self.height])

   
    def displayUpdate(self):
        # This sets the width and height of each grid location
        
        # This sets the margin between each cell
        for changexy in self.change_list:
                row = changexy[0]
                column = changexy[1]
                color = white
                if self.map[row][column] == self.v_goal:
                    color = red
                if self.map[row][column] == self.v_node:
                    color = purple
                elif self.map[row][column] == self.v_covered:
                    color = green
                elif self.map[row][column] == self.v_obstacle:
                    color = black
                elif self.map[row][column] > 0 and self.map[row][column] <= 1.1:
                    color = (0,0,255 - int(self.map[row][column] * 255))
                pygame.draw.rect(self.screen,color,\
                                [(self.margin+self.width)*column+self.margin,\
                                 (self.margin+self.height)*row+self.margin,self.width,self.height])
        # clear change list
        del self.change_list[:]
        
        # Limit to 50 frames per second
        self.clock.tick(50)
        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()