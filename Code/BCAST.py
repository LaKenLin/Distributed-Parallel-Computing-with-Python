import sys; sys.path.append("/home/jovyan/erda_mount/__dag_config__/python3")
import numpy as np
from mpi4py import MPI


variables = ['rho', 'Etot', 'Px', 'Py']

# ---------------Boundaries and sending them--------------------
def boundary_init(rho, Etot, Px, Py, n, ghost):
    """
    called by: sending function
    returns:   dictionary containing all boundary condition arrays for one process 
               (so for left, top, ... boundaries)
    """
    boundary = {'left':   {'rho':np.empty((n, ghost)), 'Etot':np.empty((n, ghost)), 
                           'Px':np.empty((n, ghost)), 'Py':np.empty((n, ghost))},
                'top':    {'rho':np.empty((ghost, n)), 'Etot':np.empty((ghost, n)), 
                           'Px':np.empty((ghost, n)), 'Py':np.empty((ghost, n))},
                'right':  {'rho':np.empty((n, ghost)), 'Etot':np.empty((n, ghost)), 
                           'Px':np.empty((n, ghost)), 'Py':np.empty((n, ghost))},
                'bottom': {'rho':np.empty((ghost, n)), 'Etot':np.empty((ghost, n)), 
                           'Px':np.empty((ghost, n)), 'Py':np.empty((ghost, n))}}
    i=0
    for var in [rho.copy(), Etot.copy(), Px.copy(), Py.copy()]:
        boundary['left'][variables[i]][:]   = var[:, ghost:2*ghost]
        boundary['top'][variables[i]][:]    = var[ghost:2*ghost, :]
        boundary['right'][variables[i]][:]  = var[:,-2*ghost:-ghost]
        boundary['bottom'][variables[i]][:] = var[-2*ghost:-ghost,:]
        i+=1
    return boundary
    
def sending(comm, l, ghost):
    """
    called by: each process, time evolution loop
    returns:   - (just completes sending operation of boundary conditions for each 
               process)
    """
    boundary = boundary_init(l.rho,l.Etot,l.Px,l.Py, l.n, ghost)
    sides = ['left', 'top', 'right', 'bottom']
    tagg = 0  # defining tag as many similar packages are sent
    for side in sides:
        for variable in variables:
            comm.Send(boundary[side][variable][:], dest=l.receivers[side], tag=tagg)    
            tagg += 1

def receiving(comm, l, ghost):
    """
    called by: each process, time evolution loop
    returns:   each of the variables with their new boundary conditions as received
               from the other processes
    """
    boundary = {'left':   {'rho':np.empty((l.n, ghost)), 'Etot':np.empty((l.n, ghost)), 
                           'Px':np.empty((l.n, ghost)), 'Py':np.empty((l.n, ghost))},
                'top':    {'rho':np.empty((ghost, l.n)), 'Etot':np.empty((ghost, l.n)), 
                           'Px':np.empty((ghost, l.n)), 'Py':np.empty((ghost, l.n))},
                'right':  {'rho':np.empty((l.n, ghost)), 'Etot':np.empty((l.n, ghost)), 
                           'Px':np.empty((l.n, ghost)), 'Py':np.empty((l.n, ghost))},
                'bottom': {'rho':np.empty((ghost, l.n)), 'Etot':np.empty((ghost, l.n)), 
                           'Px':np.empty((ghost, l.n)), 'Py':np.empty((ghost, l.n))}}
    sides = ['right', 'bottom', 'left', 'top']  # mirroring sending order
    tagg = 0
    for side in sides:
        for variable in variables:
            comm.Recv(boundary[side][variable][:], source=l.receivers[side], tag=tagg)
            tagg += 1
    i=0
    for var in [l.rho, l.Etot, l.Px, l.Py]:
        var[:, 0:ghost]  = boundary['left'][variables[i]][:]
        var[0:ghost, :]  = boundary['top'][variables[i]][:]
        var[:, -ghost:] = boundary['right'][variables[i]][:]
        var[-ghost:, :] = boundary['bottom'][variables[i]][:]
        i+=1
    return l

def gathering(comm, rank, size, u, l, root, n_local, axis_divide, ghost):
    """
    called by: each process, root process is sending & receiving, other processes 
               just sending
    returns:   class u updated and put together after the local processes have done 
               their jobs.
    """
    sendbuf_rho  = l.rho[ghost:-ghost,ghost:-ghost]
    sendbuf_Etot = l.Etot[ghost:-ghost,ghost:-ghost]
    sendbuf_Px   = l.Px[ghost:-ghost,ghost:-ghost]
    sendbuf_Py   = l.Py[ghost:-ghost,ghost:-ghost]
    master_rho  = comm.gather(sendbuf_rho, root=0)
    master_Etot = comm.gather(sendbuf_Etot, root=0)
    master_Px   = comm.gather(sendbuf_Px, root=0)
    master_Py   = comm.gather(sendbuf_Py, root=0)
    if rank == 0:
        for ran in range(size):   # looping over ranks
            pos_i = np.floor_divide(ran, axis_divide)
            pos_j = ran%axis_divide
            u.rho[pos_i*n_local:(pos_i+1)*n_local, pos_j*n_local:(pos_j+1)*n_local]  = master_rho[ran].copy()
            u.Etot[pos_i*n_local:(pos_i+1)*n_local, pos_j*n_local:(pos_j+1)*n_local] = master_Etot[ran].copy()
            u.Px[pos_i*n_local:(pos_i+1)*n_local, pos_j*n_local:(pos_j+1)*n_local]   = master_Px[ran].copy()
            u.Py[pos_i*n_local:(pos_i+1)*n_local, pos_j*n_local:(pos_j+1)*n_local]   = master_Py[ran].copy()
    return u
