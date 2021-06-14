import sys; sys.path.append("/home/jovyan/erda_mount/__dag_config__/python3")
import numpy as np
from HLL import *
from HD_BLAST import *

class local(hd):
    """
    class for localizing from global mesh according to process rank
    """
    t = 0.0
    def __init__(self, u, n_local, gamma, rank, axis_divide, global_n, ghost, setup):
        # calculating processor position within setup
        self.pos_i       = np.floor_divide(rank, axis_divide)
        self.pos_j       = rank%axis_divide
        # finding limits of sub-mesh within global mesh
        self.find_limits(n_local, global_n, ghost)
        # finding receivers/sources of boundary conditions
        self.find_receivers(setup)
        
        self.n     = n_local+2*ghost   # size of local mesh incl. ghost zones
        
        # initializing rho and Etot arrays
        self.rho   = np.ones((self.n, self.n))
        self.Etot  = np.ones((self.n, self.n))
        # copying relevant part from main mesh
        self.rho[ghost:-ghost, ghost:-ghost] = u.rho[self.limits['top']:self.limits['bottom'], self.limits['left']:self.limits['right']].copy()
        self.Etot[ghost:-ghost, ghost:-ghost]= u.Etot[self.limits['top']:self.limits['bottom'], self.limits['left']:self.limits['right']].copy()
        # initializing left boundary condition
        if self.edge_L:
            self.rho[ghost:-ghost, 0:ghost]       = u.rho[self.limits['top']:self.limits['bottom'], self.limits['left']-ghost:].copy()
            self.Etot[ghost:-ghost, 0:ghost]      = u.Etot[self.limits['top']:self.limits['bottom'], self.limits['left']-ghost:].copy()
        else:
            self.rho[ghost:-ghost, 0:ghost]       = u.rho[self.limits['top']:self.limits['bottom'], self.limits['left']-ghost:self.limits['left']].copy()
            self.Etot[ghost:-ghost, 0:ghost]      = u.Etot[self.limits['top']:self.limits['bottom'], self.limits['left']-ghost:self.limits['left']].copy()
        # initializing right boundary condition
        if self.edge_R: 
            self.rho[ghost:-ghost, -ghost:]  = u.rho[self.limits['top']:self.limits['bottom'], 0:ghost].copy()
            self.Etot[ghost:-ghost, -ghost:]  = u.Etot[self.limits['top']:self.limits['bottom'], 0:ghost].copy()
        else: 
            self.rho[ghost:-ghost, -ghost:]  = u.rho[self.limits['top']:self.limits['bottom'], self.limits['right']+1:self.limits['right']+1+ghost].copy()
            self.Etot[ghost:-ghost, -ghost:]  = u.Etot[self.limits['top']:self.limits['bottom'], self.limits['right']+1:self.limits['right']+1+ghost].copy()
        # initializing top boundary condition
        if self.edge_T:
            self.rho[0:ghost, ghost:-ghost]       = u.rho[self.limits['top']-ghost:, self.limits['left']:self.limits['right']].copy()
            self.Etot[0:ghost, ghost:-ghost]      = u.Etot[self.limits['top']-ghost:, self.limits['left']:self.limits['right']].copy()
        else:
            self.rho[0:ghost, ghost:-ghost]       = u.rho[self.limits['top']-ghost:self.limits['top'], self.limits['left']:self.limits['right']].copy()
            self.Etot[0:ghost, ghost:-ghost]      = u.Etot[self.limits['top']-ghost:self.limits['top'], self.limits['left']:self.limits['right']].copy()
        # initializing bottom boundary condition
        if self.edge_B: 
            self.rho[-ghost:, ghost:-ghost]  = u.rho[0:ghost, self.limits['left']:self.limits['right']].copy()
            self.Etot[-ghost:, ghost:-ghost]  = u.Etot[0:ghost, self.limits['left']:self.limits['right']].copy()
        else: 
            self.rho[-ghost:, ghost:-ghost]  = u.rho[self.limits['bottom']+1:self.limits['bottom']+1+ghost, self.limits['left']:self.limits['right']].copy()
            self.Etot[-ghost:, ghost:-ghost]  = u.Etot[self.limits['bottom']+1:self.limits['bottom']+1+ghost, self.limits['left']:self.limits['right']].copy()
        #initializing remaining variables
        self.Px    = np.zeros((self.n, self.n))
        self.Py    = np.zeros((self.n, self.n))
        self.gamma = u.gamma
        self.x     = u.x[self.limits['left']:self.limits['right']]
        self.y     = u.y[self.limits['top']:self.limits['bottom']]
        self.ds    = self.x[1]-self.x[0]
        self.dx    = self.ds
        self.r = np.zeros((n_local, n_local))
        for i in range(n_local):
            self.r[i] = (self.y ** 2 + self.x[i] ** 2) ** 0.5
            
    def find_limits(self, n_local, global_n, ghost):
        """
        called by: each process
        returns:   the indices from which (L,T) to which (R,B) the local array 
                   is cut out of the global array. Boolian indicating edge position
                   within global mesh.
        """
        self.limits = {}
        self.limits['left'] = int(self.pos_j*n_local)
        self.limits['right'] = int((self.pos_j+1)*n_local)
        self.limits['top'] = int(self.pos_i*n_local)
        self.limits['bottom'] = int((self.pos_i+1)*n_local)
        self.edge_L, self.edge_T, self.edge_R, self.edge_B = False, False, False, False
        if self.limits['right']+ghost >= global_n: 
            self.edge_R = True
        if self.limits['bottom']+ghost >= global_n: 
            self.edge_B = True
        if self.limits['left']-ghost <0: 
            self.edge_L = True
        if self.limits['top']-ghost <0: 
            self.edge_T = True
    
    def find_receivers(self, setup):
        """
        called by: each process
        returns:   receivers/sources of boundary conditions (cell on top, to the left, ...)
        """
        self.receivers = {}
        self.receivers['left'] = setup[self.pos_i, self.pos_j-1]
        self.receivers['top']  = setup[self.pos_i-1, self.pos_j]
        if self.edge_R: 
            self.receivers['right'] = setup[self.pos_i, 0]
        else: 
            self.receivers['right'] = setup[self.pos_i, self.pos_j+1]
        if self.edge_B: 
            self.receivers['bottom'] = setup[0, self.pos_j]
        else: 
            self.receivers['bottom'] = setup[self.pos_i+1, self.pos_j]