import sys
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None    
import matplotlib.pyplot as plt
import argparse
# implement weno5 scheme from Chi-Wang Shu paper
# https://apps.dtic.mil/sti/tr/pdf/ADA390653.pdf
#
# extend for non-uniform curvilinear grids
# perform checks for isentropic vortex on a wavy grid
# and ensure 5th order accuracy
#
def get_xp(arr):
    return cp if (cp is not None and isinstance(arr, cp.ndarray)) else np

def xp_from(arr):
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np

def weno5_fd_2d(F, js, je, axis=1):
    """
    5th-order FD-WENO reconstruction (left/right) for 3D array along given axis.
    
    Parameters
    ----------
    F : xp.ndarray
        3D array of cell-centered values (nfields, nx, ny)
    js, je : int
        Interior indices for reconstruction
        Requires 3 ghost cells on each side
        js >= 3 and je <= nx-4
    axis : int
        Axis along which to reconstruct (1=rows/xi, 2=columns/eta)
    
    Returns
    -------
    FL, FR : xp.ndarray
        Left and right reconstructed values at midpoints
    """
    xp = xp_from(F)
    Fv = xp.moveaxis(F, axis, 1)
    FL = xp.zeros_like(Fv)
    FR = xp.zeros_like(Fv)
    nvars, n, m = Fv.shape
    
    i = xp.arange(js-1, je)    
    eps = 1e-6
    
    # Shifted arrays for stencil
    Fmm = Fv[:,i-2, :]
    Fm  = Fv[:,i-1, :]
    F0  = Fv[:,i, :]
    Fp  = Fv[:,i+1, :]
    Fpp = Fv[:,i+2, :]
    Fppp= Fv[:,i+3, :]
    
    # Candidate derivatives (left)
    d0 = (2*Fmm - 7*Fm + 11*F0)/6.0
    d1 = (-Fm + 5*F0 + 2*Fp)/6.0
    d2 = (2*F0 + 5*Fp - Fpp)/6.0
    beta0 = 13/12*(Fmm - 2*Fm + F0)**2 + 1/4*(Fmm - 4*Fm + 3*F0)**2
    beta1 = 13/12*(Fm - 2*F0 + Fp)**2 + 1/4*(Fm - Fp)**2
    beta2 = 13/12*(F0 - 2*Fp + Fpp)**2 + 1/4*(3*F0 - 4*Fp + Fpp)**2
    alpha0 = 0.1 / (eps + beta0)**2
    alpha1 = 0.6 / (eps + beta1)**2
    alpha2 = 0.3 / (eps + beta2)**2
    wsum = alpha0 + alpha1 + alpha2
    w0, w1, w2 = alpha0/wsum, alpha1/wsum, alpha2/wsum
    FL[:,i, :] = w0*d0 + w1*d1 + w2*d2
    
    # Candidate derivatives (right)
    d0 = (11*Fp -7*Fpp + 2*Fppp)/6.0
    d1 = (2*F0 + 5*Fp - Fpp)/6.0
    d2 = (-Fm + 5*F0 + 2*Fp)/6.0
    beta0 = 13/12*(Fp - 2*Fpp + Fppp)**2 + 1/4*(3*Fp - 4*Fpp + Fppp)**2
    beta1 = 13/12*(F0 - 2*Fp + Fpp)**2 + 1/4*(F0 - Fpp)**2
    beta2 = 13/12*(Fm - 2*F0 + Fp)**2 + 1/4*(Fm - 4*F0 + 3*Fp)**2
    alpha0 = 0.1 / (eps + beta0)**2
    alpha1 = 0.6 / (eps + beta1)**2
    alpha2 = 0.3 / (eps + beta2)**2
    wsum = alpha0 + alpha1 + alpha2
    w0, w1, w2 = alpha0/wsum, alpha1/wsum, alpha2/wsum
    FR[:,i, :] = w0*d0 + w1*d1 + w2*d2
    
    FL = xp.moveaxis(FL, 1, axis)
    FR = xp.moveaxis(FR, 1, axis)
    return FL, FR

def linear_weno_2d(X, js, je, axis=1):
    """
    5th-order linear WENO reconstruction for 2D array along given axis.
    
    Parameters
    ----------
    X : xp.ndarray
        2D array of cell-centered values
    js, je : int
        Interior indices for reconstruction
    axis : int
        Axis along which to reconstruct (1=rows/xi, 0=columns/eta)
    
    Returns
    -------
    Xint : xp.ndarray
        2D array of reconstructed midpoints along specifjed axis
    """
    # Move axis to 0 for easjer vectorized indexing
    xp = xp_from(X)
    Xv = xp.moveaxis(X, axis, 0)  # shape (n, m)
    Xint = xp.zeros_like(Xv)
    n, m = Xv.shape
    i = xp.arange(js-2, je+1)
    # factors to blend forward and backward weno differences
    # because the first and last point cannot do backward and
    # and forward with fringe available, FD fluxes need those
    fac_f = xp.ones_like(i)*0.5
    fac_b = xp.ones_like(i)*0.5
    fac_b[0]=0.0;fac_f[0]=1.0
    fac_b[-1]=1.0; fac_f[-1]=0
    # Shifted arrays for WENO stencil (vectorized)
    Xmm = Xv[i-2, :]
    Xm  = Xv[i-1, :]
    X0  = Xv[i, :]
    Xp  = Xv[i+1, :]
    Xpp = Xv[i+2, :]
    Xppp= Xv[(i+3)%n, :]
    d0 = (2*Xmm - 7*Xm + 11*X0)/6.0
    d1 = (-Xm + 5*X0 + 2*Xp)/6.0
    d2 = (2*X0 + 5*Xp - Xpp)/6.0
    w0, w1, w2 = 0.1, 0.6, 0.3
    Xint[i, :] = fac_b[:,xp.newaxis]*(w0*d0 + w1*d1 + w2*d2)
    d0 = (11*Xp - 7*Xpp + 2*Xppp)/6
    d1 = (2*X0 + 5*Xp - Xpp)/6
    d2 = (-Xm + 5*X0 + 2*Xp)/6
    Xint[i,:] += fac_f[:,xp.newaxis]*(w0*d0 + w1*d1 + w2*d2)
    # Move axis back and divide by number of contributions
    Xint = xp.moveaxis(Xint, 0, axis)
    return Xint

def fd_metrics(x, y, js, je, ks, ke):
    """
    Compute Finite Difference metrics at nodes

    Parameters
    ----------
    x, y : 2D xp.ndarray
        Physical coordinates of the grid nodes.
    js, je, ks, ke : int
        Start and end indices in j and k directions (Python indexing).

    Returns
    -------
    xi_x, xi_y : 2D xp.ndarray
        ξ metrics at nodes
    eta_x, eta_y : 2D xp.ndarray
        η metrics at nodes
    area : 2D xp.ndarray
        area metrics computed as a Jacobian (not used in residual)
    """
    xp = xp_from(x)
    n, m = x.shape
    # --- Reconstruct midpoints ---
    xint_xi  = linear_weno_2d(x, js, je, axis=0)
    yint_xi  = linear_weno_2d(y, js, je, axis=0)
    xint_et = linear_weno_2d(x, ks, ke, axis=1)
    yint_et = linear_weno_2d(y, ks, ke, axis=1)

    # --- Compute metrics at interior points and one
    # --- extra point on either side to average for interfaces
    i = xp.arange(js-1, je+1)
    j = xp.arange(ks-1, ke+1)
    
    dx_dxi  = xint_xi[i,ks-1:ke+1] - xint_xi[i-1,ks-1:ke+1]
    dy_dxi  = yint_xi[i,ks-1:ke+1] - yint_xi[i-1,ks-1:ke+1]
    dx_det  = xint_et[js-1:je+1,j] - xint_et[js-1:je+1,j-1]
    dy_det  = yint_et[js-1:je+1,j] - yint_et[js-1:je+1,j-1]
    J = dx_dxi*dy_det - dx_det*dy_dxi

    xi_x = dy_det/J
    xi_y = -dx_det/J
    eta_x = -dy_dxi/J
    eta_y = dx_dxi/J

    metrics={ "xi_x" : xi_x, "xi_y" :xi_y, "eta_x" : eta_x, "eta_y" :eta_y, "area" : J}
    return metrics

def fv_metrics(x, y, js, je, ks, ke, check_closure=True):
    """
    Compute FV face normals and cell areas on a 2D curvilinear grid.

    Parameters
    ----------
    x, y : 2D xp.ndarray
        Physical coordinates of the grid nodes.
    js, je, ks, ke : int
        Start and end indices in j and k directions (Python indexing).
    linear_weno_2d : function
        Function to compute linear WENO reconstruction along a given axis.
    check_closure : bool, default=True
        If True, prints the max deviation from discrete closure.

    Returns
    -------
    xi_x, xi_y : 2D xp.ndarray
        ξ-face (East) scaled normals.
    eta_x, eta_y : 2D xp.ndarray
        η-face (North) scaled normals.
    area : 2D xp.ndarray
        Cell areas computed consistently with face normals.
    """
    xp = xp_from(x)
    # --- Step 1: reconstruct corner coordinates ---
    nx,ny = x.shape
    x_corner = linear_weno_2d(x, js, je, axis=0)
    x_corner = linear_weno_2d(x_corner, ks, ke, axis=1)
    y_corner = linear_weno_2d(y, js, je, axis=0)
    y_corner = linear_weno_2d(y_corner, ks, ke, axis=1)

    # --- Step 3: compute ξ and η scaled face normals ---
    xi_x = y_corner[js-1:je, ks:ke] - y_corner[js-1:je, ks-1:ke-1]   # East
    xi_y = -(x_corner[js-1:je, ks:ke] - x_corner[js-1:je, ks-1:ke-1])
    eta_x = -(y_corner[js:je, ks-1:ke] - y_corner[js-1:je-1, ks-1:ke])  # North
    eta_y = (x_corner[js:je, ks-1:ke] - x_corner[js-1:je-1, ks-1:ke])

    # --- Step 4: extract cell corners ---
    x00 = x_corner[js-1:je-1,ks-1:ke-1]
    x10 = x_corner[js:je, ks-1:ke-1]
    x01 = x_corner[js-1:je-1,ks:ke]
    x11 = x_corner[js:je, ks:ke]
    y00 = y_corner[js-1:je-1,ks-1:ke-1]
    y10 = y_corner[js:je, ks-1:ke-1]
    y01 = y_corner[js-1:je-1,ks:ke]
    y11 = y_corner[js:je, ks:ke]


    # --- Step 5: compute face midpoints ---
    xS = 0.5*(x00 + x10);  yS = 0.5*(y00 + y10)
    xE = 0.5*(x10 + x11);  yE = 0.5*(y10 + y11)
    xN = 0.5*(x11 + x01);  yN = 0.5*(y11 + y01)
    xW = 0.5*(x01 + x00);  yW = 0.5*(y01 + y00)
    
    # debug
    #ii=nx//2+1
    #jj=ny//2+1
    #plt.plot(x[ii,jj],y[ii,jj],'ro')
    #i1=ii-js
    #j1=jj-ks
    #plt.plot([x00[i1,j1],x10[i1,j1]],[y00[i1,j1],y10[i1,j1]],'k')
    #plt.plot([x10[i1,j1],x11[i1,j1]],[y10[i1,j1],y11[i1,j1]],'k')
    #plt.plot([x11[i1,j1],x01[i1,j1]],[y11[i1,j1],y01[i1,j1]],'k')
    #plt.plot([x01[i1,j1],x00[i1,j1]],[y01[i1,j1],y00[i1,j1]],'k')    
    #plt.gca().quiver(xE[i1,j1],yE[i1,j1],xi_x[i1+1,j1],xi_y[i1+1,j1],scale=5)
    #plt.gca().quiver(xW[i1,j1],yW[i1,j1],-xi_x[i1,j1],-xi_y[i1,j1],scale=5)
    #plt.gca().quiver(xN[i1,j1],yN[i1,j1],eta_x[i1,j1+1],eta_y[i1,j1+1],scale=5)
    #plt.gca().quiver(xS[i1,j1],yS[i1,j1],-eta_x[i1,j1],-eta_y[i1,j1],scale=5)
    #plt.plot([xE[i1,j1],xE[i1,j1]+xi_x[i1+1,j1]],[yE[i1,j1],yE[i1,j1]+xi_y[i1+1,j1]],'m-')
    #plt.plot([xW[i1,j1],xW[i1,j1]-xi_x[i1-1,j1]],[yW[i1,j1],yW[i1,j1]-xi_y[i1-1,j1]],'m-')
    #plt.plot([xW[i1,j1],xW[i1,j1]-xi_x[i1-1,j1]],[yW[i1,j1],yW[i1,j1]-xi_y[i1-1,j1]],'m-')
    #plt.show()
    
    # --- Step 6: compute cell areas via divergence theorem ---
    area = 0.5 * (
          xE*xi_x[1:,  :]  + yE*xi_y[1:,   :]   # East
        - xW*xi_x[:-1, :]  - yW*xi_y[:-1,  :]   # West
        + xN*eta_x[:, 1:]  + yN*eta_y[:,  1:]   # North
        - xS*eta_x[:, :-1] - yS*eta_y[:, :-1]   # South
    )

    # --- Step 7: optional closure check ---
    if check_closure:
        # discrete divergence of face normals should be ~0
        cx = ( xi_x[1:, :] - xi_x[:-1, :] ) + ( eta_x[:, 1:] - eta_x[:, :-1] )
        cy = ( xi_y[1:, :] - xi_y[:-1, :] ) + ( eta_y[:, 1:] - eta_y[:, :-1] )
        print("Closure check: max |cx| = {:.3e}, max |cy| = {:.3e}".format(xp.max(xp.abs(cx)), xp.max(xp.abs(cy))))

    metrics={ "xi_x" : xi_x, "xi_y" :xi_y, "eta_x" : eta_x, "eta_y" :eta_y, "area" : area}
    return metrics

# -----------------------------
# Flux functions
# -----------------------------
def flux_x(U, gamma=1.4):
    xp = xp_from(U)
    rho = U[0]; u = U[1]/rho; v = U[2]/rho; E = U[3]
    p = (gamma-1)*(E - 0.5*rho*(u**2 + v**2))
    F = xp.array([rho*u, rho*u**2 + p, rho*u*v, (E+p)*u])
    return F

def flux_y(U, gamma=1.4):
    xp = xp_from(U)
    rho = U[0]; u = U[1]/rho; v = U[2]/rho; E = U[3]
    p = (gamma-1)*(E - 0.5*rho*(u**2 + v**2))
    G = xp.array([rho*v, rho*u*v, rho*v**2 + p, (E+p)*v])
    return G

def max_wave_speed(U, gamma=1.4):
    xp = xp_from(U)
    rho = U[0]; u = U[1]/rho; v = U[2]/rho; E = U[3]
    p = (gamma-1)*(E - 0.5*rho*(u**2 + v**2))
    c = xp.sqrt(gamma*p/rho)
    return xp.max(xp.abs(u) + c), xp.max(xp.abs(v) + c)

def interface_wave_speed(UL, UR, n_x, n_y, js, je, ks, ke, gamma=1.4, offx=-1, offy=0):
    """
    Compute contravariant Lax-Friedrichs wave speed at interfaces along a given axis.
    Returns
    -------
    alpha : xp.ndarray
        Contravariant wave speed at interfaces in the given slice.
    """
    # --- Slice interior interfaces ---
    xp = xp_from(UL)
    rho_avg = 0.5*(UL[0][js+offx:je, ks+offy:ke] + UR[0][js+offx:je, ks+offy:ke])
    u_avg   = 0.5*(UL[1][js+offx:je, ks+offy:ke] + UR[1][js+offx:je, ks+offy:ke]) / rho_avg
    v_avg   = 0.5*(UL[2][js+offx:je, ks+offy:ke] + UR[2][js+offx:je, ks+offy:ke]) / rho_avg
    E_avg   = 0.5*(UL[3][js+offx:je, ks+offy:ke] + UR[3][js+offx:je, ks+offy:ke])
    # --- Pressure and sound speed ---
    p_avg = (gamma-1)*(E_avg - 0.5*rho_avg*(u_avg**2 + v_avg**2))
    c = xp.sqrt(gamma*p_avg / rho_avg)
    # --- Contravariant velocity along interface ---
    V = u_avg*n_x + v_avg*n_y
    # --- Metric magnitude ---
    metric_mag = xp.sqrt(n_x**2 + n_y**2)
    # --- Lax-Friedrichs wave speed ---
    alpha = xp.abs(V) + c*metric_mag
    return alpha

def reconstruct(U, js, je, gamma=1.4, axis=1):
    xp = xp_from(U)
    # --- Create Cartesian fluxes
    F = flux_x(U, gamma)  # x-direction flux
    G = flux_y(U, gamma)  # y-direction flux    
    # --- Reconstruct interface fluxes and field values ---
    FL, FR = weno5_fd_2d(F, js, je, axis=axis)
    GL, GR = weno5_fd_2d(G, js, je, axis=axis)
    UL, UR = weno5_fd_2d(U, js,je,axis=axis)
    
    return FL, FR, GL, GR, UL, UR

    
def residual_fv(U, js, je, ks, ke, metrics, gamma=1.4):
    """
    Compute 2D WENO5 Finite Volume residual for Euler equations on curvilinear grids.
    Vectorized and interior-only.
    
    Parameters
    ----------
    U : xp.ndarray, shape (4, nx, ny)
        Conserved variables [rho, rho*u, rho*v, E], including ghost cells
    x, y : xp.ndarray, shape (nx, ny)
        Physical coordinates, including ghost cells
    js, je, ks, ke : int
        Interior index ranges
    gamma : float
        Ratio of specific heats
    
    Returns
    -------
    Res : xp.ndarray, shape (4, nx, ny)
        Residuals at interior cells (ghosts remain 0)
    """
    # init residual
    xp = xp_from(U)
    Res = xp.zeros_like(U)

    # fetch metrics
    xi_x = metrics["xi_x"]
    xi_y = metrics["xi_y"]
    eta_x= metrics["eta_x"]
    eta_y= metrics["eta_y"]
    area = metrics["area"]

    # reconstruct j+1/2, k values
    FL_xi, FR_xi, GL_xi, GR_xi, UL_xi, UR_xi = reconstruct(U, js, je, gamma, axis=1)
    # wave-speed in contravariant coordinate xi for dissipation scaling
    alpha_xi=interface_wave_speed(UL_xi, UR_xi, xi_x, xi_y, js, je, ks, ke, offx=-1, offy=0)
    # FV flux in xi direction
    F_xi = 0.5*(FL_xi[:,js-1:je,ks:ke] + FR_xi[:,js-1:je,ks:ke])*xi_x + \
           0.5*(GL_xi[:,js-1:je,ks:ke] + GR_xi[:,js-1:je,ks:ke])*xi_y + \
           - 0.5*alpha_xi*(UR_xi[:,js-1:je,ks:ke] - UL_xi[:,js-1:je,ks:ke])
    
    #reconstruct j, k+1/2 values
    FL_eta, FR_eta, GL_eta, GR_eta, UL_eta, UR_eta = reconstruct(U, ks, ke, gamma, axis=2)
    # wave-speed in contravariant coordinate xi for dissipation scaling    
    alpha_eta = interface_wave_speed(UL_eta, UR_eta, eta_x, eta_y, js, je , ks,ke, offx=0, offy=-1)
    # FV flux in eta direction
    F_et = 0.5*(FL_eta[:,js:je,ks-1:ke] + FR_eta[:,js:je,ks-1:ke])*eta_x + \
           0.5*(GL_eta[:,js:je,ks-1:ke] + GR_eta[:,js:je,ks-1:ke])*eta_y + \
           -0.5*alpha_eta*(UR_eta[:,js:je,ks-1:ke] - UL_eta[:,js:je,ks-1:ke])
    # finite volume residual -sum(F.n)/A
    Res[:, js:je,ks:ke] =  -((F_xi[:,1:,:] - F_xi[:,:-1,:])  \
                             +(F_et[:,:,1:]- F_et[:,:,:-1]))/area
    # apply periodicity
    apply_periodic(Res)
    return Res


def residual_fd(U, js, je, ks, ke, metrics, gamma=1.4):
    """
    Compute 2D WENO5 Finite Difference residual for Euler equations on curvilinear grids.
    Vectorized and interior-only.
    
    Parameters
    ----------
    U : xp.ndarray, shape (4, nx, ny)
        Conserved variables [rho, rho*u, rho*v, E], including ghost cells
    x, y : xp.ndarray, shape (nx, ny)
        Physical coordinates, including ghost cells
    js, je, ks, ke : int
        Interior index ranges
    gamma : float
        Ratio of specific heats
    

    Returns
    -------
    Res : xp.ndarray, shape (4, nx, ny)
        Residuals at interior cells (ghosts remain 0)
    """
    xp = xp_from(U)
    Res = xp.zeros_like(U)
    # fetch metrics
    xi_x = metrics["xi_x"]
    xi_y = metrics["xi_y"]
    eta_x= metrics["eta_x"]
    eta_y= metrics["eta_y"]
    
    # --- Create Cartesian Fluxes
    F = flux_x(U, gamma)  # x-direction flux
    G = flux_y(U, gamma)  # y-direction flux
    
    # reconstruct j+1/2, k values
    FL_xi, FR_xi, GL_xi, GR_xi, UL_xi, UR_xi = reconstruct(U, js, je, gamma, axis=1)
    # wave-speed in contravariant coordinate xi for dissipation scaling
    xi_x_avg = (xi_x[:-1,1:-1]+xi_x[1:,1:-1])*0.5
    xi_y_avg = (xi_y[:-1,1:-1]+xi_y[1:,1:-1])*0.5    
    alpha_xi=interface_wave_speed(UL_xi, UR_xi, xi_x_avg, xi_y_avg, js, je, ks, ke, offx=-1, offy=0)
    #divide by xi_x_avg assuming xi is the dominant direction to physical x coord
    alpha_xi = alpha_xi/xi_x_avg
    
    # fd xi flux
    # add dissipation term only to xi flux (again assuming xi is dominant along x)    
    F_xi = 0.5*(FL_xi[:,js-1:je,ks:ke] + FR_xi[:,js-1:je,ks:ke]) \
        - 0.5*alpha_xi*(UR_xi[:,js-1:je,ks:ke] - UL_xi[:,js-1:je,ks:ke])

    G_xi = 0.5*(GL_xi[:,js-1:je,ks:ke] + GR_xi[:,js-1:je,ks:ke]) 

    # reconstruct j, k+1/2 values
    FL_eta, FR_eta, GL_eta, GR_eta, UL_eta, UR_eta = reconstruct(U, ks, ke, gamma, axis=2)
    # wave-speed in contravariant coordinate xi for dissipation scaling
    eta_x_avg = (eta_x[1:-1,:-1]+eta_x[1:-1,1:])*0.5
    eta_y_avg = (eta_y[1:-1,:-1]+eta_y[1:-1,1:])*0.5    
    alpha_eta=interface_wave_speed(UL_eta, UR_eta, eta_x_avg, eta_y_avg, js, je, ks, ke, offx=0, offy=-1)
    #divide by eta_x_avg assuming eta is the dominant direction to physical y coord
    alpha_eta = alpha_eta/eta_y_avg

    # fd eta flux
    F_et = 0.5*(FL_eta[:,js:je,ks-1:ke] + FR_eta[:,js:je,ks-1:ke])
    # add dissipation term only to eta flux (again assuming eta is dominant along y)
    G_et = 0.5*(GL_eta[:,js:je,ks-1:ke] + GR_eta[:,js:je,ks-1:ke]) \
        - 0.5*alpha_eta*(UR_eta[:,js:je,ks-1:ke] - UL_eta[:,js:je,ks-1:ke])

    # full strong form of the finite difference residual at the solution
    # nodes (or cell centers)
    # F_x + G_y = F_xi * xi_x + F_eta * eta_x + G_xi * xi_y + G_eta * eta_y
    Res[:, js:je,ks:ke] =  -(F_xi[:,1:,:] - F_xi[:,:-1,:])*xi_x[1:-1,1:-1] \
                           -(G_xi[:,1:,:] - G_xi[:,:-1,:])*xi_y[1:-1,1:-1] \
                           -(F_et[:,:,1:]- F_et[:,:,:-1])*eta_x[1:-1,1:-1] \
                           -(G_et[:,:,1:]- G_et[:,:,:-1])*eta_y[1:-1,1:-1]
    apply_periodic(Res)
    return Res

# -----------------------------
# RK3 time integration
# -----------------------------
def rk3_tvd(U0, dt, js, je, ks, ke, metrics, residual, gamma=1.4):
    xp = xp_from(U0)
    R0 = residual(U0, js, je, ks, ke, metrics, gamma)
    U1 = U0 + dt*R0
    R1 = residual(U1, js, je, ks, ke, metrics, gamma)
    U2 = 0.75*U0 + 0.25*(U1 + dt*R1)
    R2 = residual(U2, js, je, ks, ke, metrics, gamma)
    U3 = (U0 + 2*(U2 + dt*R2)) / 3.0
    return U3, xp.linalg.norm(U3-U0)

# -----------------------------
# Boundary conditions (periodic)
# -----------------------------
def apply_periodic(U, ng=3):
    U[:, :ng, :] = U[:, -2*ng:-ng, :]
    U[:, -ng:, :] = U[:, ng:2*ng, :]
    U[:, :, :ng] = U[:, :, -2*ng:-ng]
    U[:, :, -ng:] = U[:, :, ng:2*ng]

# -----------------------------
# Isentropic vortex initial condition
# -----------------------------
def init_isentropic_vortex(nx, ny, x0=0.0, y0=0.0, Lx=10.0, Ly=10.0, u0=0.5, v0=0, gamma=1.4):
    ng = 3
    xp = cp if (cp is not None) else np
    x = xp.linspace(-Lx/2, Lx/2, nx)
    y = xp.linspace(-Ly/2, Ly/2, ny)
    XX, YY = xp.meshgrid(x, y, indexing='ij')
    # make a wavy grid
    X = XX + 0.1*xp.sin(YY)
    Y = YY + 0.1*xp.sin(XX)
    beta = 1.0
        
    r2 = (X-x0)**2 + (Y-y0)**2
    u = u0 - beta/(2*xp.pi)*(Y-y0)*xp.exp(0.5*(1-r2))
    v = v0 + beta/(2*xp.pi)*(X-x0)*xp.exp(0.5*(1-r2))
    T = 1.0 - ((gamma-1)*beta**2)/(8*gamma*xp.pi**2)*xp.exp(1-r2)
    rho = T**(1/(gamma-1))
    p = rho**(gamma)
    E = p/(gamma-1) + 0.5*rho*(u**2 + v**2)
    
    U = xp.zeros((4,nx,ny))
    U[0] = rho; U[1] = rho*u; U[2] = rho*v; U[3] = E
    return U, X, Y

# -----------------------------
# Plotting function
# -----------------------------
def plot_density(XX, YY, U, title='Density'):
    """
    Contour plot of density fjeld
    """
    n,m = XX.shape
    xp = xp_from(XX)
    if xp!=np:
      X = xp.asnumpy(XX[3:n-3,3:m-3])
      Y = xp.asnumpy(YY[3:n-3,3:m-3])
    else:
      X = XX[3:n-3,3:m-3]
      Y = YY[3:n-3,3:m-3]
    rho = U[0][3:n-3,3:m-3]
    plt.figure(figsize=(6,5))
    ccp = plt.contourf(X, Y, rho, levels=50, cmap='viridis')
    plt.colorbar(ccp)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.plot(X,Y,'k--',linewidth=0.1)
    plt.plot(X.T,Y.T,'k--',linewidth=0.1)
    plt.axis('equal')
    plt.tight_layout()

def plot_density_line(X, U, Uf, title='Density'):
    xp = xp_from(U)
    if xp!=np:
      rho = xp.asnumpy(U[0])
      rhof = xp.asnumpy(Uf[0])
      XX = xp.asnumpy(X)
    else:
      rho = U[0]
      rhof= Uf[0]
      XX = X
    plt.figure(figsize=(6,5))
    n,m = X.shape
    print(n,m)
    print(U.shape)
    plt.plot(XX[:,m//2+1],rho[:,m//2+1],'r.-')
    plt.plot(XX[:,m//2+1],rhof[:,m//2+1],'bo-')
    plt.show()
    
# -----------------------------
# Main routine
# -----------------------------
def main(restype):
    # choose appropriate metrics and residual routines
    if restype == 'fv':
        compute_metrics = fv_metrics
        residual = residual_fv
    else:
        compute_metrics = fd_metrics
        residual = residual_fd
    # Grid parameters
    nx, ny = 41, 41        # grid points
    Lx, Ly = 10.0, 10.0    # physical domain
    CFL = 5.0              # CFL number
    gamma = 1.4
    u0=0.5
    # Initialize
    U, X, Y = init_isentropic_vortex(nx, ny, Lx=Lx, Ly=Ly, u0=u0, gamma=gamma)
    xp = xp_from(U)
    # find distance to travel to return back to initial location periodically
    # (interior domain size + dx)/u0
    t_final = (X[nx-4,ny//2+1]-X[3,ny//2+1])/u0 + (X[1,ny//2+1]-X[0,ny//2+1])/u0
    print("t_final=",t_final)
    # Plot initial density
    plot_density(X, Y, U, title='Initial Density')    
    # Time stepping loop
    t = 0.0
    dt = 0.1  # initial time step; can compute CFL-based later
    js, je = 3, nx-3
    ks, ke = 3, ny-3
    metrics = compute_metrics(X,Y,js,je,ks,ke)
    dx = (X[1,ny//2+1]-X[0,ny//2+1])
    print(f' mean_area {xp.mean(metrics["area"])} {dx*dx}')
    #t_final = 0.1
    
    while t < t_final:
        # Apply periodic BCs
        # Compute max wave speeds for CFL-based dt
        t += dt
        alpha_x, alpha_y = max_wave_speed(U[:,js:je,ks:ke], gamma)
        dt_cfl = CFL * min( (X[1,0]-X[0,0])/xp.max(alpha_x),
                            (Y[0,1]-Y[0,0])/xp.max(alpha_y) )
        dt = min(dt, t_final-t, dt_cfl)        
        # Advance one time step with RK3-TVD
        U, dUnorm = rk3_tvd(U, dt, js, je, ks, ke, metrics, residual, gamma)
        print(f"time :{t:0.3f} {dUnorm:0.3f}")
    
    # Plot final density
    plot_density(X, Y, U, title=f'Density at t={t_final:.3f}')
    plt.show()
# -----------------------------
# Main routine for checking error
# ----------------------------- 
def errorcheck(restype):
    max_p = 8
    N0 = 10
    if restype == 'fv':
        compute_metrics = fv_metrics
        residual = residual_fv
    else:
        compute_metrics = fd_metrics
        residual = residual_fd
    inc = 1.5
    for p in range(1,max_p+1): 
        ny =  int(N0 * inc**(p-1))+1
        nx =  2 * int(N0 * inc**(p-1))+1
        Lx, Ly = 20, 10   # physical domain
        CFL = 1.0              # CFL number
        t_final = 1.0         # final time
        gamma = 1.4
        # Initialize
        U, X, Y = init_isentropic_vortex(nx, ny, Lx=Lx, Ly=Ly, gamma=gamma)
        Uf,Xf,Yf = init_isentropic_vortex(nx, ny, x0=t_final*0.5, Lx=Lx, Ly=Ly, gamma=gamma)
        # Plot initial density
        #plot_density(X, Y, U, title='Initial Density')
        dx = Lx/(nx-1)
        # Time stepping loop
        t = 0.0
        dt = 0.1  # initial time step; can compute CFL-based later
        js, je = 3, nx-3
        ks, ke = 3, ny-3
        metrics = compute_metrics(X,Y,js,je,ks,ke)
        xp = xp_from(U) 
        while t < t_final:
            # Compute max wave speeds for CFL-based dt
            alpha_x, alpha_y = max_wave_speed(U[:,js:je,ks:ke], gamma)
            dt_cfl = CFL * min( (X[1,0]-X[0,0])/xp.max(alpha_x),
                                (Y[0,1]-Y[0,0])/xp.max(alpha_y) )
            dt = min(dt, t_final-t, dt_cfl)
        
            # Advance one time step with RK3-TVD
            U, dUnorm = rk3_tvd(U, dt, js, je, ks, ke, metrics, residual, gamma)
            t += dt
            print(f"time :{t:0.3f} {dUnorm:0.3f}")
    
            # Plot final density
            #plot_density(X, Y, U, title=f'Density at t={t_final:.3f}')
        err = xp.sqrt(xp.mean(xp.square(U[0,js:je,ks:ke]-Uf[0,js:je,ks:ke])))
        # Plot final density
        plot_density_line(X, U, Uf, title=f'Density at t={t_final:.3f}')
        if p > 1:
            print(f"dx: {dx:0.5f} Convergence order: {xp.log(errprev/err)/xp.log(inc):.3f}, Error: {err:.3e}")
        else:
            print(f"dx: {dx:0.5f} Error: {err:.3e}")
        errprev=err

def parse_arguments(args):
    """
    Parses command line arguments, specifically --value=some_value
    """
    parser = argparse.ArgumentParser(description="A simple parser for a --value argument.")
    
    # Add the expected argument
    # 'dest="my_value"' specifies the attribute name in the resulting Namespace object
    parser.add_argument(
        '--residual', 
        dest='residual', 
        type=str,  # Expect a string value
        help='A required value for the script, e.g., --residual FV',
        default = 'FV',
        required=False # Makes the argument mandatory
    )
    parser.add_argument(
        '--execution', 
        dest='execution', 
        type=str,  # Expect a string value
        help='A required value for the script, e.g., --execution ERRORCHECK',
        default = 'MAIN',
        required=False # Makes the argument mandatory
    )

    # Parse the arguments
    args = parser.parse_args()
    
    # The 'args' object will always have 'args.my_value' defined, 
    # either from the user input or the default.
    print(f"Using residual : {args.residual}")
    print(f"Running        : {args.execution}")
    
    return vars(args)

# -----------------------------
# Run the solver
# -----------------------------
if __name__ == "__main__":
    options = parse_arguments(sys.argv[1:])
    if options["execution"].lower() == 'main':
        main(options["residual"].lower())
    else:
        errorcheck(options["residual"].lower())
