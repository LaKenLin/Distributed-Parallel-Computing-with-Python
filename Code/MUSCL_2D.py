import numpy as np
from HLL import *
from HD_BLAST import *

# -------------TIME-STEPPING SCHEME--------------------
def muscl_2d(u, dt, Slope=MonCen, Riemann_Solver=HLL):
    dtds = dt / u.ds

    # 1) Compute primitive variables (rho, v, P)
    # make sure to copy u.rho to D, otherwise D becomes a "pointer" to u.rho, and updates itself in 5)
    D = np.copy(u.rho)  # D: density
    v = u.velocity()  # v: velocity
    P = u.pressure()  # P: pressure

    # 2) Compute slopes based on centered points
    dD = Slope(D)  # returns (2,n,n)
    dP = Slope(P)  # returns (2,n,n)
    dv0 = Slope(v[0])  # returns (2,n,n)
    dv1 = Slope(v[1])  # returns (2,n,n)
    dv = np.array([dv0, dv1])  # shape = (2,2,n,n)

    # 3) Trace forward to find solution at [t+dt/2, x +- dx/2]
    # Time evolution with source terms
    div = dv[0, 0] + dv[1, 1]  # div(v)
    D_t = - v[0] * dD[0] - v[1] * dD[1] - D * div  # dD/dt = -v*grad(D) - D*div(v)
    P_t = - v[0] * dP[0] - v[1] * dP[1] - u.gamma * P * div  # dP/dt = -v*grad(P) - gamma*P*div(v)
    v_t = np.copy(v)  # shape = (2,n,n)
    v_t[0] = - v[0] * dv[0, 0] - v[1] * dv[0, 1] - dP[0] / D  # dv/dt = -v*grad(v) - grad(P)/D
    v_t[1] = - v[0] * dv[1, 0] - v[1] * dv[1, 1] - dP[1] / D

    # Loop over X- and Y-direction
    for axis in range(2):
        s = void()

        # Calculates the parallel and perpendicular velocities to the coordinate direction
        if axis == 0:  # X-axis
            # Parallel (U) and perpendicular (V) velocities
            U = v[0]; dU = dv[0]; U_t = v_t[0]
            V = v[1]; dV = dv[1]; V_t = v_t[1]

        if axis == 1:  # Y-axis
            # Parallel (U) and perpendicular (V) velocities
            U = v[1]; dU = dv[1]; U_t = v_t[1]
            V = v[0]; dV = dv[0]; V_t = v_t[0]

        # 4) Spatial interpolation + time terms in X-direction

        # left state -- AS SEEN FROM THE INTERFACE
        s.Dl = D + 0.5 * dD[axis] + 0.5 * dtds * D_t
        s.Pl = P + 0.5 * dP[axis] + 0.5 * dtds * P_t
        s.Ul = U + 0.5 * dU[axis] + 0.5 * dtds * U_t
        s.Vl = V + 0.5 * dV[axis] + 0.5 * dtds * V_t

        # right state -- AS SEEN FROM THE INTERFACE
        s.Dr = D - 0.5 * dD[axis] + 0.5 * dtds * D_t
        s.Pr = P - 0.5 * dP[axis] + 0.5 * dtds * P_t
        s.Ur = U - 0.5 * dU[axis] + 0.5 * dtds * U_t
        s.Vr = V - 0.5 * dV[axis] + 0.5 * dtds * V_t

        # 5) Roll down -1 so that we collect left and right state (as seen from the interface) at the same grid index
        # right state, at left state interface
        s.Dr = np.roll(s.Dr, -1, axis=axis)
        s.Pr = np.roll(s.Pr, -1, axis=axis)
        s.Ur = np.roll(s.Ur, -1, axis=axis)
        s.Vr = np.roll(s.Vr, -1, axis=axis)

        # 6) Solve for horizontal flux based on interface values
        flux = Riemann_Solver(s, u)

        # Update conserved variables with horizontal fluxes
        u.rho -= dtds * (flux.D - np.roll(flux.D, 1, axis=axis))
        u.Etot -= dtds * (flux.E - np.roll(flux.E, 1, axis=axis))
        if axis == 0:  # X-axis
            u.Px -= dtds * (flux.U - np.roll(flux.U, 1, axis=axis))
            u.Py -= dtds * (flux.V - np.roll(flux.V, 1, axis=axis))
        if axis == 1:  # Y-axis
            u.Px -= dtds * (flux.V - np.roll(flux.V, 1, axis=axis))
            u.Py -= dtds * (flux.U - np.roll(flux.U, 1, axis=axis))

    return u