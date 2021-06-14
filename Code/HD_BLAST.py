import numpy as np

# -------------HYDRODYNAMICS & BLAST-WAVE CLASS--------
class void():
    """ Empty class, used to create ad hoc objects """
    pass
class hd(void):
    """ Template class for hydrodynamics, with parameters, variables and methods """
    gamma = 1.4

    def __init__(self, n=64):
        """ Initialization of arrays for conservative variables """
        self.n = n
        self.rho = np.ones((n, n))
        self.Etot = np.ones((n, n))
        self.Px = np.zeros((n, n))
        self.Py = np.zeros((n, n))
        self.coordinates()

    def coordinates(self):
        """ Coordinate initialization """
        n = self.n
        self.ds = 2.0 / n
        self.dx = self.ds
        self.Lbox = n * self.ds
        self.x = self.ds * (np.arange(n) - n / 2 + 0.5)
        self.y = self.x
        self.r = np.zeros((n, n))
        for i in range(n):
            self.r[i] = (self.y ** 2 + self.x[i] ** 2) ** 0.5

    def velocity(self):
        """ Compute velocity from conservative variables """
        return np.array([self.Px / self.rho, self.Py / self.rho])

    def pressure(self):
        """ Compute pressure from conservative variables """
        if (self.gamma == 1.0):
            P = self.cs ** 2 * self.rho
        else:
            Eint = self.Etot - 0.5 * (self.Px ** 2 + self.Py ** 2) / self.rho
            P = (self.gamma - 1.) * Eint
        return P

    def temperature(self):
        """ Compute the 'temperature', defined as P/rho """
        P = self.pressure()
        return P / self.rho

    def Courant(self, C=0.5):
        """ Courant condition for HD """
        P = self.pressure()
        v = self.velocity()
        V = np.sqrt(v[0] ** 2 + v[1] ** 2)
        cs = np.sqrt(self.gamma * P / self.rho)
        return C * self.ds / np.max(cs + V)
    
class blast_wave(hd):
    """ An extension of the hd() class with initial conditions """
    t = 0.0
    def __init__(u,n=64,gamma=1.4,e0=1e3,d0=1.0,power=2,w=3.,eps=0.01):
        hd.__init__(u,n)
        u.power = power
        u.w = w*u.ds
        u.gamma = gamma
        factor = (3./w)**2
        u.rho  = d0*np.ones((n,n))
        u.Etot = 1.0 + e0*np.exp(-(u.r/u.w)**power)

class blast_wave_mass(hd):
    """ An extension of the hd() class with initial conditions """
    t = 0.0
    def __init__(u,n=64,gamma=1.4,e0=1e4,d0=0.5e3,power=2,w=3.,eps=0.01):
        hd.__init__(u,n)
        u.power = power
        u.w = w*u.ds
        u.gamma = gamma
        factor = (3./w)**2
        u.rho  = 1.0 + d0*np.exp(-(u.r/u.w)**power)
        u.Etot = 1.0 + e0*np.exp(-(u.r/u.w)**power)

class blast_wave_mass_noise(hd):
    """ An extension of the hd() class with initial conditions """
    t = 0.0
    def __init__(u,n=64,gamma=1.4,e0=1e3,d0=0.5e3,power=2,w=3.,eps=0.01):
        hd.__init__(u,n)
        u.power = power
        u.w = w*u.ds
        u.gamma = gamma
        factor = (3./w)**2
        u.rho  = 1.0 + d0*np.exp(-(u.r/u.w)**power) + d0*eps*np.random.rand(n,n)
        u.Etot = 1.0 + e0*np.exp(-(u.r/u.w)**power)
        
class blast_wave_collision(hd):
    """ An extension of the hd() class with initial conditions """
    t = 0.0
    def __init__(u,n=64,gamma=1.4,e0=1e5,d0=1e3,power=4,w=3.,eps=0.01):
        hd.__init__(u,n)
        ds = 2.0/n
        x = ds*(np.arange(n)-n/2+0.5)
        y = x
        r_plus = np.zeros((n,n))
        r_minus = np.zeros((n,n))
        shift = 10
        for i in range(n):
            r_plus[i] = ((y-ds*n/shift)**2 + (x[i]-ds*n/shift)**2)**0.5
            r_minus[i] = ((y+ds*n/shift)**2 + (x[i]+ds*n/shift)**2)**0.5
        u.power = power
        u.w = w*ds
        u.gamma = gamma
        factor = (3./w)**2
        u.rho  = 1.0 + d0*np.exp(-(r_plus/u.w)**power) + d0*np.exp(-(r_minus/u.w)**power)+ d0*eps*np.random.rand(n,n)
        u.Etot = 1.0 + e0*np.exp(-(r_plus/u.w)**power) + e0*np.exp(-(r_minus/u.w)**power)
        
