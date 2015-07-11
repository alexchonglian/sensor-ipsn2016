'''
Created on Oct 1, 2012

@author: imaveek
'''
import numpy as np

class Radio(object):
    '''
    Radio provides RToF measurements
    '''

    def __init__(self, sf):
        '''
        Constructor
        '''
        self.sf = sf
        self.noise = sf.noise_radio
        
        
    def rss(self, anchors):
        '''
        Model for distance to RToF measurements
        '''
        diff = np.array([(np.array(self.sf.xy) - np.array(anchor.xy)) for anchor in anchors])
        d = np.sum(np.abs(diff)**2,axis=-1)**(1./2)
        d_noisy = np.random.normal(d, self.noise)
        return d_noisy