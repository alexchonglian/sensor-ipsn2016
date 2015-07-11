'''
Created on 2012-10-13

@author: Administrator
'''

import numpy as np
from Graph import goalgraph
from PIL import Image

class Case(object):
    '''
    classdocs
    '''


    def __init__(self, name, arena_bitmap_path = None):
        '''
        Constructor
        '''
        self.name = name
        
        if (arena_bitmap_path == None):
            self.arena_bitmap_path = "./ArenaMaps/empty.bmp"
        else:
            self.arena_bitmap_path = arena_bitmap_path
        im = Image.open(self.arena_bitmap_path)  # @UndefinedVariable
        self.al = im.size[0]
        self.aw = im.size[1]
        self.map_array = np.asarray(im).astype(float)
        
        self.num_explorers = 4
        self.num_anchors = 6
        self.max_iterations = 1
        self.deltick = 1
        
        self.noise_velocity = 0.1
        self.noise_turn = 0.1
        self.noise_radio = 0.1
        self.noise_mag = 40
        self.fail_prob = 0
        
        self.rf_signature_db = {}
        
        self.num_total_run = 1000
        self.num_particles = 100
        self.init_p_wt = 1/float(self.num_particles)
        self.cov_algo = "drunkwalk"
                
        self.start = []
        self.end = []
        self.goal_graph = None
        
        self.stop_on_all_covered = False
        self.is_display_on_real = False
        
    def reset(self):
        if self.goal_graph is not None:
            self.goal_graph = goalgraph.GoalGraph()