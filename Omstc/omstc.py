'''
Created on 2012-10-4

@author: Administrator
'''
from ostcontrol import OSTControl

class Omstc(object):
    '''
    Omstc multi-agent coverage algorithm
    '''

    def __init__(self, sflist):
        '''
        Constructor
        '''
                
        # Initialize attributes
        self.control = OSTControl(sflist)        

    def command(self, sflist, cMap):
        '''
        Command as per the coverage algorithm
        '''
        # Give the coverage command
        self.control.command(sflist, cMap)