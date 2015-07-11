'''
Created on Feb 12, 2013

@author: imaveek
'''
from libc.math cimport sqrt

def sigdist(sig1,sig2):
        '''
        Gives distance between RToF signatures
        '''
        cdef float s1
        cdef float s2
        n = len(sig1)
        cdef float sum = 0
        for i in range(n):
            s1 = sig1[i]
            s2 = sig2[i]
            sum = sum + ((s1 - s2)**2)
        d = sqrt(sum)
        return d