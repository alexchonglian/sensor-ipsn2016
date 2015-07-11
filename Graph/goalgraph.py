'''
Created on Oct 8, 2013

@author: imaveek
'''
import scipy.io
import numpy as np
import networkx as nx
from numpy import append

class GoalGraph(object):
    '''
    classdocs
    '''
    def __init__(self, ):
        '''
        Constructor
        '''
        
#         # Get start and goal locations extracted from map
#         self.start = self.__getPixelsFromMat('./ArenaMaps/13room_slam_map/v1.mat')
#         self.goal_dirs = [90,0,0,0,0,270,270,0,0,180,90,0,180]
#         # Build a dictionary of pixels to node
#           
#         node_names = ['./ArenaMaps/13room_slam_map/v1.mat', './ArenaMaps/13room_slam_map/v2.mat', \
#                      './ArenaMaps/13room_slam_map/v3.mat', './ArenaMaps/13room_slam_map/v4.mat', \
#                      './ArenaMaps/13room_slam_map/v5.mat', './ArenaMaps/13room_slam_map/v6.mat', \
#                      './ArenaMaps/13room_slam_map/v7.mat','./ArenaMaps/13room_slam_map/v8.mat', \
#                      './ArenaMaps/13room_slam_map/v9.mat','./ArenaMaps/13room_slam_map/v10.mat',\
#                      './ArenaMaps/13room_slam_map/v11.mat','./ArenaMaps/13room_slam_map/v12.mat',\
#                      './ArenaMaps/13room_slam_map/v13.mat'
#                      ]
#           
#         # Create a connectivity graph
#         self.graph = nx.Graph()
#         self.graph.add_nodes_from([0,1,2,3,4,5,6,7,8,9,10,11,12])
#         self.graph.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(4,6),(6,7),(6,10),\
#                                    (7,9),(7,8),(8,10),(10,11),(11,12)])        
        
        
        # Get start and goal locations extracted from map
        self.start = self.__getPixelsFromMat('./ArenaMaps/7room_slam_map_small_new/v1.mat')
        self.goal_dirs = [180, 90, 0, 270, 180, 270, 180]
        # Build a dictionary of pixels to node
           
        node_names = ['./ArenaMaps/7room_slam_map_small_new/v1.mat', './ArenaMaps/7room_slam_map_small_new/v2.mat', \
                     './ArenaMaps/7room_slam_map_small_new/v3.mat', './ArenaMaps/7room_slam_map_small_new/v4.mat', \
                     './ArenaMaps/7room_slam_map_small_new/v5.mat', './ArenaMaps/7room_slam_map_small_new/v6.mat', \
                     './ArenaMaps/7room_slam_map_small_new/v7.mat']
           
        # Create a connectivity graph
        self.graph = nx.Graph()
        self.graph.add_nodes_from([0,1,2,3,4,5,6])
        self.graph.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(2,5)])

#         # Get start and goal locations extracted from map
#         self.start = self.__getPixelsFromMat('./ArenaMaps/6room_map_small/v1.mat')
#         self.goal_dirs = [90, 0, 0, 270, 0, 0]
#         # Build a dictionary of pixels to node
#          
#         node_names = ['./ArenaMaps/6room_map_small/v1.mat', './ArenaMaps/6room_map_small/v2.mat', \
#                      './ArenaMaps/6room_map_small/v3.mat', './ArenaMaps/6room_map_small/v4.mat', \
#                      './ArenaMaps/6room_map_small/v5.mat', './ArenaMaps/6room_map_small/v6.mat']
#          
#         # Create a connectivity graph
#         self.graph = nx.Graph()
#         self.graph.add_nodes_from([0,1,2,3,4,5])
#         self.graph.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(2,5)])
        
        
        self.node_id_by_cell = {}
        self.node_cells_by_id = {}
        self.num_nodes = len(node_names)
        
        self.nodes_covered = [False] * self.num_nodes
        for i in range(len(node_names)):
            node_pixels = self.__getPixelsFromMat(node_names[i])
            self.node_cells_by_id[i] = node_pixels
            # node_pixels = [[p[0], p[1]] for p in node_pixels]
            for p in node_pixels:
                self.node_id_by_cell[tuple(p)] = i
            if (i == 0):
                continue
    
    
    def getCellsFromId(self, node_id):
        return self.node_cells_by_id.get(node_id, None);
    
    
    def markCovered(self, loc):
        # Mark coverage based on real location
        r_id = self.node_id_by_cell.get(tuple(loc), None)
        if (r_id != None):
            self.nodes_covered[r_id] = True
    
    
    def getCoverage(self):
        is_all_covered = all(self.nodes_covered)
        pct_covered = sum(self.nodes_covered) / float(self.num_nodes)
        return [pct_covered, is_all_covered]
    
    
    def getNextDir(self, est_cell):
        '''
        Get next location based on current location in graph
        '''
        node_id = self.node_id_by_cell.get(tuple(est_cell), None);        
        if (node_id != None):
            return self.goal_dirs[node_id]
        return None
    
    
    def getNodeFromCell(self, cell):
        node_id = self.node_id_by_cell.get(tuple(cell), None);
        return node_id
        
    
    def getNextDirSample(self, est_cells):
        '''
        Get next location based on current location in graph
        '''
        # Find the node_id of the
        node_id_counts = np.zeros(self.num_nodes, np.float32)
        for est_cell in est_cells:
            node_id = self.node_id_by_cell.get(tuple(est_cell.tolist()), None);
            if node_id is not None:
                node_id_counts[node_id] += 1.0  
        # if any node is contained sample the node_id
        if node_id_counts.any():
            max_prob_node = np.argmax(node_id_counts)
            max_node_nbors = self.graph.neighbors(max_prob_node)
            node_id_counts[max_node_nbors] = np.max(node_id_counts) * 0.25
            
            # normalize node id counts
            # modified
            print np.sum(node_id_counts)
            print self.goal_dirs
            norm1 = [ 0.80000000, 0.2, 0, 0, 0, 0, 0]   #modified from 0.8000000001
        
            return np.random.choice(self.goal_dirs, 1, p = norm1)
            
            # return self.goal_dirs[chosen_node_id]
        return None      
    
                
    def __getPixelsFromMat(self, filepath):
        mat = scipy.io.loadmat(filepath)
        pix = mat['pixels']
        return pix
    