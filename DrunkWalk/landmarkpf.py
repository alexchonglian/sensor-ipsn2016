'''
Created on Sep 30, 2013

@author: imaveek
'''
import numpy as np
import scipy.stats as stats
import cresample as resample

class LandmarkPF(object):
    '''
    Class for particle filter based location estimation
    '''
    
    # Constants
    COVARIANCE = 1
    PF_CORRECT_THRESH_K = 2
    
    def __init__(self, init_pf):
        self.num_particles = init_pf.num_particles
        self.particles_xy = np.copy(init_pf.particles_xy)
        self.weights = np.copy(init_pf.weights)
        self.cov_vector = [self.COVARIANCE] * self.num_particles
        
    def normalizeWts(self):
        sum_wts = np.sum(self.weights)
        
        if (sum_wts == 1): return self.weights
        
        if sum_wts != 0:
            self.weights = self.weights/sum_wts
        else:
            self.weights.fill(1/float(self.num_particles))
        return  self.weights
        
        
    def getEstimate(self):
        '''
        Estimate location from particles_xy
        '''
        den = sum(self.weights)
        x = sum(self.particles_xy[:,0] * self.weights) / den
        y = sum(self.particles_xy[:,1] * self.weights) / den
        return (x,y)

    
    def predict(self):
        '''
        Predict the location for each landmark particle
        '''
        pass
    
    
    def correct(self, robot_pf):
        '''
        Correct the weight for each particle
        '''
        pos_estimate = robot_pf.getEstimate()
        diff = self.particles_xy - pos_estimate
        distances = np.sum(np.abs(diff)**2,axis=-1)**(1./2)
        # Generate a normal distribution with mean as distance and a constant STD
        wt_dists = stats.norm(distances, self.cov_vector)
        # Find the probability of the distance between K and -K
        self.weights = wt_dists.cdf(self.PF_CORRECT_THRESH_K) - wt_dists.cdf(-self.PF_CORRECT_THRESH_K)



    def resample_fast(self):
        u0 = np.random.random()
        self.normalizeWts()
        indices = resample.cresample(self.weights, u0)
        self.particles_xy = self.particles_xy[indices,:]
        self.weights.fill(1/float(self.num_particles))   
        

    def resample(self):
        '''
        Resample the particle distribution based on the weights
        '''
        indices = []
        self.weights = self.normalizeWts()
        C = [0.] + [sum(self.weights[:i+1]) for i in range(self.num_particles)]
        u0, j = np.random.random(), 0
        for u in [(u0 + i)/self.num_particles for i in range(self.num_particles)]:
            while u > C[j]:
                j += 1
            indices.append(j-1)
        self.particles_xy = self.particles_xy[indices,:]
        self.weights.fill(1/float(self.num_particles))    
