'''
Created on Feb 12, 2013

@author: imaveek
'''
import random
import numpy as np
import csigdist as cs
     
class Landmark:
    '''
    Defines a landmark
    '''
    # Constants
    SIG_PROXIMITY_THRESH = 5

    def __init__(self, sig):
        self.sig = sig
        self.hash = str(random.random() * 1000)
        
    def isRevisit(self, sig):
        '''
        RToF signature proximity function
        '''
        d = cs.sigdist(self.sig, sig)
#        print "================================ d = " + str(d)
        if (d < self.SIG_PROXIMITY_THRESH):
            return True
        else:
            return False
        
    def sigdist(self,sig1,sig2):
        '''
        Gives distance between RToF signatures
        '''
        d = np.linalg.norm(sig1-sig2, 2)
        return d