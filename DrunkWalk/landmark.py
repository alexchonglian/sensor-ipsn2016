'''
Created on Feb 12, 2013

@author: imaveek
'''
import random
import numpy as np
from landmarkpf import LandmarkPF

class LandmarkDB:
    '''
    Defines a central database of detected landmarks
    '''
    
    def __init__(self, case):
        self.lm_count = 0
        self.landmarks = {}
        self.noise_radio = case.noise_radio

        
    def match(self, signature, pose_pf):
        '''
        Add signature to the database and assign initial position estimate
        '''
        # Check if it is a revisited landmark
        for _, lm in self.landmarks.items():
            if (lm.isRevisit(signature)):
                lm.pf.correct(pose_pf)
                lm.pf.resample_fast()
                return lm.pf
        # if it is not a revisit add new landmark
        self.lm_count += 1
        lm = Landmark(self.lm_count, signature, pose_pf, self.noise_radio)
        self.landmarks[lm.id] = lm
        return None


     
class Landmark:
    '''
    Defines a landmark
    '''
    # Constants
    

    def __init__(self, lm_count, signature, init_pf, noise_radio):
        self.id = lm_count
        self.signature = np.array(signature)
        self.hash = str(random.random() * 1000)
        self.pf = LandmarkPF(init_pf)
        self.SIG_PROXIMITY_THRESH = noise_radio
        
    def isRevisit(self, signature):
        '''
        RToF signature proximity function
        '''
        d = self.sigdist(self.signature, np.array(signature))
        if (d < self.SIG_PROXIMITY_THRESH):
            return True
        else:
            return False
        
    def sigdist(self,sig1,sig2):
        '''
        Gives distance between RToF signatures
        '''
        d = np.linalg.norm(sig1-sig2, 2) / len(sig1)
        return d
        