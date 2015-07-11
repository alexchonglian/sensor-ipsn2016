'''
Created on Sep 27, 2012

@author: imaveek
'''

from odometer import Odometer
from radio import Radio
from magnetometer import Magnetometer
import math
import random
import numpy as np
import copy

class SensorFly(object):
    '''
    SensorFly node
    '''
    def __init__(self, name, init_xy, init_dir, init_battery, \
                 noise_vel, noise_turn, noise_radio, noise_mag, fail_prob, controller, case=None):
        '''
        Params: length in no. of cells
                breadth in no. of cells
        '''
        self.name = name
        
        self.xy = [float(i) for i in init_xy] # In meters
        self.dir = float(init_dir)
        self.mag_dir = float(init_dir)
        
        self.is_alive = True 
        self.is_moving = False
        self.is_backing_off = False
        self.has_turned = False
        
        self.command = Command(0,0,0)
        self.last_command = Command(0,0,0)
        self.__real_command = Command(0,0,0)
        self.command_time_cnt = 0
        
        self.noise_velocity = float(noise_vel)
        self.noise_turn = float(noise_turn)
        self.noise_radio = float(noise_radio)
        self.noise_mag = float(noise_mag)
        self.noise_fingerprint = [[self.noise_radio/2,0],[0,self.noise_radio/2]]

        
        self.odometer = Odometer(self.noise_velocity, self.noise_turn)
        self.radio = Radio(self)
        self.mag = Magnetometer(self)
        
        self.has_collided = False
        self.is_goal_reached = False
        self.fail_prob = fail_prob
        self.collision_count = 0
        
        # Records estimated location
        self.pf_estimated_xy = np.zeros([1, 2], dtype=np.float32)
        self.dr_estimated_xy = np.zeros([1, 2], dtype=np.float32)
        
        self.pf_particles_xy = np.ones([100, 2], dtype=float) * self.xy
        
        self.sig_match_cnt = 0
        
        # Add references to the central controller
        self.controller = controller
        
        self.last_goal_dir = np.zeros([1, 2], dtype=np.float32)
        self.backoff_time_cnt = 0
        
        if case is not None:
            self.sig_db = case.rf_signature_db


        
    def setMoveCommand(self, command_params):
        '''
        Params: turn - the angle to turn in degrees
                time - the time to setMoveCommand in seconds
        '''
        self.is_moving = True
        [turn, time, velocity] = command_params 
        # Set the command_params to be executed now
        self.command.time = float(time)
        self.command.turn = float(turn)
        self.command.velocity = float(velocity)
        
        self.__real_command = copy.deepcopy(self.command)
        self.__real_command.velocity = self.odometer.velocity(velocity)
        self.__real_command.turn = self.odometer.turn(turn)
        self.command_time_cnt = self.__real_command.time
        
        # Store the last executed command_params
        self.last_command = copy.deepcopy(self.__real_command)
        
        
    def update(self, deltick, arena):
        '''
        Params: deltick - the time tick in seconds
        '''
        # Move
        if self.is_moving:
            # Turn instantly in first tick
            if (self.command_time_cnt == self.__real_command.time):
                self.dir = (self.dir + self.__real_command.turn) % 360
                self.has_turned = True
            else:
                self.has_turned = False
                
            # Get the magnetometer reading for direction
            self.mag_dir = self.mag.getDirection()
                
            # Set the time taking into account the tick size    
            if (self.command_time_cnt - deltick <= 0):
                deltime = self.command_time_cnt
                self.command_time_cnt = 0
            else:
                self.command_time_cnt = self.command_time_cnt - deltick
                deltime = deltick
                
            
            oldpos = self.xy
            newx = self.xy[0] + ((self.__real_command.velocity * deltime) * \
                             math.cos(math.radians(self.dir)))
            newy = self.xy[1] + ((self.__real_command.velocity * deltime) * \
                             math.sin(math.radians(self.dir)))
            newpos = [newx, newy]
                        
            is_collision = self.isCollision(oldpos,newpos,arena)
            
            # If no collision setMoveCommand to new point
            if is_collision == False:
                self.xy = [newx, newy]
            else: # Else stop at object boundary
                self.is_moving = False
                self.command_time_cnt = 0
                self.is_backing_off = False
             
            # If the movement is complete
            if (self.command_time_cnt == 0):
                self.is_moving = False
                self.is_backing_off = False
        else:
            deltime = 0
        
    
    def isCollision(self, oldpos, newpos, arena):
        '''
        Compute if collision
        '''        
        [oldX, oldY] = arena.gridmap.xytocell(oldpos)
        [newX, newY] = arena.gridmap.xytocell(newpos)
        covCells = self.bresenhamLine(oldX, oldY, newX, newY)
        for c in covCells:
            if arena.gridmap.map[c[0]][c[1]] == arena.gridmap.v_obstacle: # hit obstacle
                self.has_collided = True
                self.collision_count += 1
                p = random.random()
                if (p < self.fail_prob):
                    self.is_alive = False
                return True
        self.has_collided = False
        return False


    def bresenhamLine(self,x,y,x2,y2):
        # Brensenham line algorithm
        steep = 0
        coords = []
        dx = abs(x2 - x)
        if (x2 - x) > 0: sx = 1
        else: sx = -1
        dy = abs(y2 - y)
        if (y2 - y) > 0: sy = 1
        else: sy = -1
        if dy > dx:
            steep = 1
            x,y = y,x
            dx,dy = dy,dx
            sx,sy = sy,sx
        d = (2 * dy) - dx
        for _ in range(0,dx):
            if steep: coords.append((y,x))
            else: coords.append((x,y))
            while d >= 0:
                y = y + sy
                d = d - (2 * dx)
            x = x + sx
            d = d + (2 * dy)
        coords.append((x2,y2))
        return coords
    
    
    def getRadioSignature(self):
        
#         anchors = [0,1,2,3,5,6]
#         signature = np.zeros(6,np.float32)
#         for i,a in enumerate(anchors):
#             [x,y] = self.xytocell(self.xy)
#             signature[i] = np.random.choice(self.sig_db[(y,x,a)])
        signature  = np.random.multivariate_normal(self.xy, self.noise_fingerprint)
#         self.radio.rss(self.controller.anchors)
        return signature
    
    def xytocell(self, xy):
        X = int(round(xy[0]))
        Y = int(round(xy[1]))
        if X < 0:
            X = 0
        elif X >= 10:
            X = 10-1
        if Y < 0:
            Y=0
        elif Y >= 7:
            Y=7-1
        return [X,Y]
    
    
#****************************************************************************************                
#     Helper Classes
#****************************************************************************************
class Command:
    '''
    Command for SensorFly nodes
    '''
    
    def __init__(self, turn, time, velocity):
        '''
        Constructor
        '''
        self.turn = turn
        self.time = time
        self.velocity = velocity
        
    @staticmethod
    def fromList((turn, time, velocity)):
        '''
        Constructor
        '''
        return Command(turn, time, velocity)