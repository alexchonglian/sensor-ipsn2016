'''
Created on Oct 3, 2012

@author: imaveek
'''

from smparticle import Particle
import math
import numpy as np
from smcontrol import SMController
from Sim.gridmap import GridMap
from landmark import Landmark


class SugarMap(object):
    '''
    SugarMap location multi-agent coverage algorithm
    '''
    
    def __init__(self, _init_coverageProbMap, num_particles, sflist):
        '''
        Constructor
        '''
        # Initialize attributes
        self.c_map = _init_coverageProbMap
        self.num_particles = num_particles
        self.numRevisit = 0
        self.numlandmarks = 0
        self.control = SMController()
        
        self.localMaps = []
        self.landmarks = {}
        
        # Initialize particle dictionary
        self.particles = {}
        sf0 = sflist[0] # the location of the 1st sensorfly in the real world is used as a relative origin
        for sf in sflist:
            # Keep a local coverage map for each SensorFly
            sensorflyOwnMap = GridMap(self.c_map.length, self.c_map.breadth, 
                               self.c_map.celllength, self.c_map.cellbreadth)
            self.localMaps.append(sensorflyOwnMap)
            plist = []
            pwts = []
            for i in range(num_particles): #@UnusedVariable
                # A list of particles for each SensorFly
                p = Particle(sf, float(1)/float(num_particles), sensorflyOwnMap, sf0)
                plist.append(p)
                pwts.append(float(1)/float(num_particles))
            self.particles[sf.name] = [plist, pwts]
    
    
    def command(self, sflist):
        '''
        Command as per the coverage algorithm
        '''
        # Give the coverage command
        self.control.command(sflist, self.particles, self.c_map)    
    
    def update(self, sflist, deltick):
        '''
        Update the SugarMap
        '''        
        # Predict particle position
        self.predict(sflist, deltick)
        # Obtain signature and apply particle weight update
        self.correct(sflist)
        # Merge coverage maps from particles
        self.mergeLocalMaps(sflist)
        # Merge coverage maps from SensorFly's
        self.mergeGlobalMaps(sflist)
        # Re-sample particles
        self.resample(sflist)
    
    
    def predict(self, sflist, deltick):
        '''
        Predict the coverage map for each particle using odometry
        '''
        for sf in sflist:
            plist = self.particles[sf.name][0]
            # Predict next coverage grid using odometry
            turn = sf.lastCommand.turn
            velocity = sf.lastCommand.velocity
            for p in plist:
                # TODO: Add noise to the odometry according to a model
                p.setMoveCommand(turn, velocity)
                p.update(sf, deltick)
            
            
    def correct(self, sflist):
        '''
        Correct the weight for each particle using RToF signatures
        '''
        # TODO: See if this function can be optimized
        for sf in sflist:
            sig = sf.locSignature
            plist = self.particles[sf.name][0]
            pwts = self.particles[sf.name][1]
            
            isRevisit = False
            if sf.name in self.landmarks:
                landmarks = self.landmarks[sf.name]
            else:
                self.landmarks[sf.name] = []
                landmarks = self.landmarks[sf.name]
                
            # Check if it is a revisited landmark and update particle weights
            if landmarks:
                for lm in landmarks:
                    if (lm.isRevisit(sig)):
                        isRevisit = True
                        self.numRevisit += 1
                        self.updateParticleWtOnRevisit(plist, pwts, lm.hash)

            
            # if it is not a revisit add new landmark
            if not isRevisit:
                lm = Landmark(sig)
                self.numlandmarks += 1
                landmarks.append(lm)
                # For each particle for a SensorFly store the estimated locations
                for p in plist:
                    p.landmarkDict[lm.hash] = [p.x, p.y]
                

    def resample(self, sflist):
        '''
        Resample the particle distribution based on the  weights
        '''
        for sf in sflist:
            plist = self.particles[sf.name][0]
            pwts = self.particles[sf.name][1]
            n = len(pwts)
            indices = []
            C = [0.] + [sum(pwts[:i+1]) for i in range(n)]
            u0, j = np.random.random(), 0
            for u in [(u0+i)/n for i in range(n)]:
                while u > C[j]:
                    j+=1
                    indices.append(j-1)
            plist = [plist[idx] for idx in indices]
            plist[-1].weight = 1.0/n
            pwts = [pwts[idx] for idx in indices]
                
    
    def mergeLocalMaps(self, sflist):
        '''
        Merge the coverage map of all particles to compute the local probabilistic coverage
        map for each SensorFly
        '''
        for sf, lmap in zip(sflist, self.localMaps):
            plist = self.particles[sf.name][0]
            lmap.map[:] = 0.0
            
            for p in plist:
                lmap.map = lmap.map + (p.localMap.map * p.weight)
                

                
    def mergeGlobalMaps(self, sflist):
        '''
        Merge the coverage map of all SensorFly nodes to compute global probabilistic coverage map
        '''
        self.c_map.map[:] = 1.0
        for lmap in self.localMaps:
            self.c_map.map *= 1 - lmap.map
        self.c_map.map = 1 - self.c_map.map
        for sf in sflist:
            plist = self.particles[sf.name][0] # get particle lists of the current sensorfly
            loc  = self.control.locEstimate(plist) # determine sensorfly's location using particles' locations
            self.c_map.markMap(loc[0],loc[1],self.c_map.vNode)

    
    
    def updateParticleWtOnRevisit(self, plist, pwts, hash_value):
        '''
        Update the weight of the particles when a revisit is identified according to a specified function
        '''
        # Update the weights in 2 passes
        # TODO: Check the weight update function
        dlist = []
        for p in plist:
            [x,y] = p.landmarkDict[hash_value]
            d = math.sqrt((p.x - x)**2 + (p.y - y)**2)
            d = math.exp(-d**2 / 9) # similarity function
            dlist.append(d)
        sumD = sum(dlist)
        # Update weights in inverse ratio of their real distance 
        for i,p in enumerate(plist):
            d = dlist[i]/sumD
            p.weight = d
            pwts[i] = p.weight
            
            
            
#        OPTIMIZATION FOR OPENCL - Add to mergeLocalMaps
#
#        ctx = cl.create_some_context()
#        queue = cl.CommandQueue(ctx) 
#        program = self.loadProgram("mergemap.cl", ctx)
#        mf = cl.mem_flags
#        #initialize client side (CPU) arrays
#        a = lmap.map
#        b = p.localMap.map
#        #create OpenCL buffers
#        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
#        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
#        dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)
#
#        program.mergemap(queue, a.shape, None, a_buf, b_buf, dest_buf)
#        c = np.empty_like(a)
#        cl.enqueue_read_buffer(queue, dest_buf, c).wait()
#        lmap.map = c

#    def loadProgram(self, filename, ctx):
#        '''
#        Read in the OpenCL source file as a string
#        '''
#        f = open(filename, 'r')
#        fstr = "".join(f.readlines())
#        #create the program
#        program = cl.Program(ctx, fstr).build()
#        return program