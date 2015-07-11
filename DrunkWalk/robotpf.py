'''
Created on Sep 30, 2013

@author: imaveek
'''

import numpy as np
import pypr.clustering.gmm as gmm
import cresample as resample
from scipy import stats


class RobotPF(object):
    '''
    Class for particle filter based location estimation
    '''
    # Constants
    COVARIANCE_MATRIX = np.diag([2,2])
        
    def __init__(self, sf, num_particles, init_wt, noise_model, deltick):
        self.sf = sf
        self.num_particles = num_particles
        self.particles_xy = np.ones([self.num_particles, 2], dtype=float) * sf.xy
        self.particles_dir = np.ones(self.num_particles, dtype=float) * sf.dir
        self.weights = np.ones(self.num_particles, dtype=float) * init_wt
        self.lm_covars = np.array([self.COVARIANCE_MATRIX for _ in range(self.num_particles)])
        self.noise_t = noise_model[0]
        self.noise_v = noise_model[1]
        self.deltick = deltick
    
    
    def getEstimate(self):
        '''
        Estimate location from particles_xy
        '''
        den = np.sum(self.weights)
        xy = (self.particles_xy * self.weights[:,np.newaxis]) / den
        return np.sum(xy,0)
    
    
    def getEntropy(self, wts):
        p_log_p = -1 * (wts * np.log(wts))
        return np.sum(p_log_p)
    
    
    def normalizeWts(self):
        sum_wts = np.sum(self.weights)
        if (sum_wts == 1): return self.weights
        if sum_wts != 0:
            self.weights = self.weights/sum_wts
        else:
            self.weights.fill(1/float(self.num_particles))
        return  self.weights
    
    
    def predict(self, cmd):
        '''
        Predict the next position for each particle using odometry
        '''
        
#         self.particles_dir = np.random.normal(self.sf.mag_dir, self.sf.noise_mag, self.num_particles)
        # Compute new poses only if it has turned       
        if self.sf.has_turned:
            # Add noise to turn
            nT = np.abs(self.sf.command.turn * self.noise_t)
            if nT == 0: nT = 0.1
            t = np.random.normal(self.sf.command.turn, nT, self.num_particles)
            # Compute new poses
            self.particles_dir = (t + self.particles_dir) % 360
        
        # add noise to velocities
        nV = np.abs(cmd.velocity * self.noise_v)
        nV = 0.1 if nV == 0.0 else nV
        vdt = np.random.normal(cmd.velocity, nV, self.num_particles) \
            * self.deltick
        # compute new position
        dx = vdt * np.cos(np.radians(self.particles_dir))
        dy = vdt * np.sin(np.radians(self.particles_dir))
        dxy = np.concatenate((dx[:,np.newaxis], dy[:,np.newaxis]), axis=1) 
        self.particles_xy = self.particles_xy + dxy

            
    def correct(self, landmark_db):
        '''
        Correct the weight for each particle using RToF signatures
        '''
        
        # Magnetometer correction
        dir_dist =  stats.norm(self.sf.mag_dir, self.sf.noise_mag)
        self.weights = dir_dist.pdf(self.particles_dir)
        # Resample
        self.resample_fast()

        # Signature correction
        signature = self.sf.getRadioSignature()
        update_pf = landmark_db.match(signature, self)        
        
        # If no signature found nothing to do
        if (update_pf == None):
            return
        # Increment signature match counts
        self.sf.sig_match_cnt += 1
        
        # Create a GMM from the update_pf
        lm_means = update_pf.particles_xy
        lm_weights = update_pf.normalizeWts()
        # Use the GMM to update weights
        new_wts = gmm.gmm_pdf(self.particles_xy, lm_means, self.lm_covars, lm_weights, False)
        
        # If weights are all 0 return
        if not np.any(new_wts):
            return
        self.weights = new_wts
        
    
    def resample_fast(self):
        u0 = np.random.random()
        self.normalizeWts()
        indices = resample.cresample(self.weights, u0)
        self.particles_xy = self.particles_xy[indices,:]
        self.particles_dir = self.particles_dir[indices]
        self.weights.fill(1/float(self.num_particles)) 


    def resample(self):
        '''
        Resample the particle distribution based on the weights
        '''
        indices = []
        self.normalizeWts()
        
        C = [0.] + [sum(self.weights[:i+1]) for i in range(self.num_particles)]
        u0, j = np.random.random(), 0
        for u in [ (u0 + i)/self.num_particles for i in range(self.num_particles)]:
            while u > C[j]:
                j += 1
            indices.append(j-1)
        self.particles_xy = self.particles_xy[indices,:]
        self.particles_dir = self.particles_dir[indices,:]
        self.weights.fill(1/float(self.num_particles))
        