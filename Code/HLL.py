import numpy as np
from HD_BLAST import *

# -------------RIEMANN SOLVER & SLOPES----------
def HLL(s, u):
    Dl = s.Dl; Ul = s.Ul; Pl = s.Pl; Vl = s.Vl
    Dr = s.Dr; Ur = s.Ur; Pr = s.Pr; Vr = s.Vr

    # momentum and sound speed for each side of interface (l==left, r==right)
    mUl = Dl * Ul
    mUr = Dr * Ur
    mVl = Dl * Vl
    mVr = Dr * Vr
    Eintl = Pl / (u.gamma - 1.0)
    Eintr = Pr / (u.gamma - 1.0)
    c_left = (u.gamma * Pl / Dl) ** 0.5
    c_right = (u.gamma * Pr / Dr) ** 0.5
    c_max = np.maximum(c_left, c_right)
    # maximum wave speeds to the left and right (guaranteed to have correct sign)
    SL = np.minimum(np.minimum(Ul, Ur) - c_max, 0)  # <= 0.
    SR = np.maximum(np.maximum(Ul, Ur) + c_max, 0)  # >= 0.

    # total energy per unit volume
    El = Eintl + 0.5 * Dl * (Ul ** 2 + Vl ** 2)
    Er = Eintr + 0.5 * Dr * (Ur ** 2 + Vr ** 2)

    # Hydro fluxes
    fDl = mUl
    fDr = mUr
    fUl = Ul * mUl + Pl
    fUr = Ur * mUr + Pr
    fVl = Ul * mVl
    fVr = Ur * mVr
    fEl = (El + Pl) * Ul
    fEr = (Er + Pr) * Ur

    # HLL flux based on wavespeeds.  The general form is
    #    (SR*F_left - SL*F_right + SL*SR *(u_right - u_left))/(SR-SL)
    # where u = (rho, rho*v, E_tot) are the conserved variables
    Flux = void()
    Flux.D = (SR * fDl - SL * fDr + SL * SR * (Dr - Dl)) / (SR - SL)
    Flux.E = (SR * fEl - SL * fEr + SL * SR * (Er - El)) / (SR - SL)
    Flux.U = (SR * fUl - SL * fUr + SL * SR * (mUr - mUl)) / (SR - SL)
    Flux.V = (SR * fVl - SL * fVr + SL * SR * (mVr - mVl)) / (SR - SL)
    return Flux
def left_slope(f,axis=0):
    return f-np.roll(f,1,axis)
def MonCen(f):
    """ Monotonized central slope limiter """
    if f.ndim==1:
        ls=left_slope(f)
        rs=np.roll(ls,-1)
        cs=np.zeros(ls.shape)
        w=np.where(ls*rs>0.0)
        cs[w]=2.0*ls[w]*rs[w]/(ls[w]+rs[w])
        return cs
    else:
        shape=np.insert(f.shape,0,f.ndim)
        slopes=np.zeros(shape)
        for i in range(f.ndim):
            ls=left_slope(f,axis=i)
            rs=np.roll(ls,-1,axis=i)
            cs=np.zeros(f.shape)
            w=np.where(ls*rs>0.0)
            cs[w]=2.0*ls[w]*rs[w]/(ls[w]+rs[w])
            slopes[i]=cs
        return slopes