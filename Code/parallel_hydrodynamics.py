import sys; sys.path.append("/home/jovyan/erda_mount/__dag_config__/python3")
import numpy as np
import matplotlib.pyplot as plt
from time import time
from mpi4py import MPI
# sys.path.append("/home/jovyan/erda_mount/AstroProject/modules")
from HD_BLAST import hd, blast_wave, blast_wave_mass, blast_wave_mass_noise, blast_wave_collision
from HLL      import *
from MUSCL_2D import muscl_2d
from PLOTTING import imshow
from LOCAL    import local
from BCAST    import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

axis_divide = int(np.sqrt(size))   # dividing axis according to number of processes
setup       = np.reshape(np.arange(size, dtype=int), (axis_divide, axis_divide))
ghost       = int(sys.argv[3])     # taking the number of ghost zones from terminal

# Defining the blast wave to study:
n  = int(sys.argv[1])       # reading resolution from terminal
n  = n+n%axis_divide        # clause to get even number of resolution processes
nt = int(sys.argv[2])       # reading number of iterations from terminal
d0 = 1e4;e0 = 1e5; C = 0.5; solver = HLL
u  = blast_wave(n=n, e0=e0, d0=d0, gamma=1.4, w=1., power=4, eps=0.01)
# different blase waves such as blast_wave_collision can be run

global_n = n                       # resolution of global mesh
n_local  = global_n//axis_divide   # resolution of local sub-mesh

#calling local class in forder to localize global mesh for processor rank
loc = local(u, n_local, gamma=1.4, rank=rank, axis_divide=axis_divide, global_n=global_n, ghost=ghost, setup=setup)

start = time()
for it in range(nt):
    dt=np.array(loc.Courant(C))
    col_dt = np.zeros(1)
    comm.Allreduce(dt, col_dt, MPI.MIN)  # using Allreduce to find minimum dt of all processes
    loc = muscl_2d(loc, col_dt[0], Slope=MonCen, Riemann_Solver=solver)  # stepping forward in time
    sending(comm, loc, ghost)            # custom function for sending ghost zones
    loc = receiving(comm, loc, ghost)    # custom function for receiving ghost zones
    loc.t += col_dt[0]
used=time()-start
# printing time of time-evolution loop
if rank == 0:
    print('{:.1f} sec, {:.2f} microseconds/update'.format(used,1e6*used/(n**2*nt)))
# custom function for gathering all sub-meshes into local class for root=0
u = gathering(comm, rank, size, u, loc, 0, n_local, axis_divide, ghost)
if rank == 0:
    temp = u.temperature()   # calculating temperature profile
    # custom plotting function for plotting density and temperature profile
    imshow(u.rho, temp, 'result', e0, d0, C, n, headline=('Density profile', 'Temperature profile'))