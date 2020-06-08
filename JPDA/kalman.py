class EKFcontrol():
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
    
    def getCurrentState(self):
        return self.x_k_k_min
    
    def predict(self):
        self.x_k_k_min = np.dot(self.F, self.x_k_k)
        self.P_k_k_min = np.dot(self.F, np.dot(self.P_k_k,self.F.T)) + self.Q
        
    def update(self,z):
        z = np.matrix(z).T
        z_res = z - np.dot(self.H, self.x_k_k_min)
        print(z_res)
        S_k = np.dot(np.dot(self.H, self.P_k_k_min), self.H.T) + self.R
        K_k = np.dot(self.P_k_k_min, self.H.T) * inv(S_k)
        self.x_k_k = self.x_k_k_min + np.dot (K_k, z_res)
        self.P_k_k = np.dot(np.eye(self.x_dim) - np.dot(K_k, self.H), self.P_k_k_min)