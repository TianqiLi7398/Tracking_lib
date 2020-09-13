import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import random
import scipy.linalg
from numpy.linalg import inv
import copy
from scipy.stats import multivariate_normal
from scipy.stats.distributions import chi2
from numpy import linalg as LA
# from matplotlib.patches import Ellipse
import math
import time


class EKFcontrol:
    # here actually a standard Linear KF is implemented 
    def __init__(self, _F,_H,_x,_P,_Q,_R):
        
        self.F=_F
        self.H=_H
        
        self.x_k_k_min=_x
        self.P_k_k_min=_P
        self.Q=_Q
        self.R=_R
        
        self.x_k_k=_x
        self.P_k_k=_P
        
        self.x_dim = _x.shape[0]
        self.z_dim = _H.shape[1]
        
        self.S_k = self.R
    
    def getCurrentState(self):
        return self.x_k_k_min
    
    def getEst(self):
        return np.dot(self.H, self.x_k_k)
    
    def predict(self):
        self.x_k_k_min = np.dot(self.F, self.x_k_k)
        self.P_k_k_min = np.dot(self.F, np.dot(self.P_k_k,self.F.T)) + self.Q
        
    def update(self,z):
        z = np.matrix(z).T

        self.z_bar = np.dot(self.H, self.x_k_k_min)
        self.z_res = z - self.z_bar
        self.S_k = np.dot(np.dot(self.H, self.P_k_k_min), self.H.T) + self.R
        # Cholesky's method for inverse
        # c = np.linalg.inv(np.linalg.cholesky(self.S_k))
        # inv_S = np.dot(c.T,c)
        inv_S = inv(self.S_k)
        K_k = np.dot(self.P_k_k_min, self.H.T) * inv_S
        self.x_k_k = self.x_k_k_min + np.dot(K_k, self.z_res)
        self.P_k_k = np.dot(np.eye(self.x_dim) - np.dot(K_k, self.H), self.P_k_k_min)


class measurement:
    def __init__(self, z, id_):
        self.value = z
        self.track = []
        self.g_ij = []
        self.table = {"track": self.track, "g_ij": self.g_ij}
        self.id = id_
        
    def inside_track(self, track_id):
        self.track.append(track_id)


class track:
    """track should have the function of initilization, confirmation, deletion, save all tracks"""
    #     initalized -> confirmed -> deleted
    #        |
    #        V
    #     abandoned
    def __init__(self, t0, id_, kf, DeletionThreshold, ConfirmationThreshold,
        memory_len = 10):
        self.id = id_
        self.measurement = []
        self.confirmed = False
        self.deleted = False     # if deleted after confirmed, this deleted is true
        self.kf = kf
        self.t0 = t0
        self.history = [1]
        self.record = {"points": [], "time_interval": []}
        self.abandoned = False   # if not get confirmed, this abandoned is true 
        self.S = self.kf.S_k
        self.pred_z = self.kf.x_k_k
        self.ConfirmationThreshold = ConfirmationThreshold
        self.DeletionThreshold = DeletionThreshold
        self.bb_box_size = [10, 10] # default size for bounding box
        self.agent_id = -1
        self.neighbor = []
        self.memory = {}
        self.memory_len = memory_len
        
    def track_memory(self, agent_id, sub_id):
        # accumulate memory with fiex horizon
        # TODO
        # Q: if we need to save memory for non-matching siuation
        # A: I don't like that.... that's just trival memories
        try:
            if len(self.memory[agent_id]) == self.memory_len:
                self.memory[agent_id].pop(0)
                self.memory[agent_id].append(sub_id)
            else:
                self.memory[agent_id].append(sub_id)
        except:
            self.memory[agent_id] = [sub_id]

    def update(self, t, kf, isObs):
        self.kf = kf
        self.kf.predict()
        self.record["points"].append(self.kf.getEst())
        
        if self.confirmed:
            # check if there is observation inside the gate
            if isObs:
                self.history.append(1)
                
            else:
                self.history.append(0)
            
            
            if len(self.history) > self.DeletionThreshold[1]:
                self.history.pop(0)
                    
            if len(self.history) == self.DeletionThreshold[1] and sum(self.history) <= self.DeletionThreshold[1] - self.DeletionThreshold[0]:
            #                         reaches the threshold to delete
                self.deletion(t)
        else:
            # check if there is observation inside the gate
            if isObs:
                self.history.append(1)
                
            else:
                self.history.append(0)
                
            # when enough observation is done, check the N/M
            if len(self.history) == self.ConfirmationThreshold[1]:
                if sum(self.history) < self.ConfirmationThreshold[0]:
                        # reaches the threshold to abandon, ow, confirm
                    self.abandoned = True
                else:
                    self.confirmation()
    
    def confirmation(self):
        self.confirmed = True
        
    
    def deletion(self, t):
        self.deleted = True
        self.record["time_interval"] = [self.t0, t] 
    
    def get_measurement(self, a_list):
        # here a_list is np.ndarray data type from np.where function
        self.measurement = a_list.tolist()
        return

class vertex:

    def __init__(self, track_id):
        # self.agent_id = agent_id
        self.track_id = track_id
        self.neighbor = []
        self.memory = []
    
