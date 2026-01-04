import numpy as np
import matplotlib.pyplot as plt
# implement weno5 scheme from Chi-Wang Shu paper
# https://apps.dtic.mil/sti/tr/pdf/ADA390653.pdf
def weno5_fd_2d(F, js, je, axis=1):
    """
    5th-order FD-WENO reconstruction (left/right) for 3D array along given axis.
    
    Parameters
    ----------
    F : np.ndarray
        3D array of cell-centered values (nfields, nx, ny)
    js, je : int
        Interior indices for reconstruction
        Requires 3 ghost cells on each side
        js >= 3 and je <= nx-4
    axis : int
        Axis along which to reconstruct (1=rows/xi, 2=columns/eta)
    
    Returns
    -------
    FL, FR : np.ndarray
        Left and right reconstructed values at midpoints
    """
    Fv = np.moveaxis(F, axis, 1)
    FL = np.zeros_like(Fv)
    FR = np.zeros_like(Fv)
    nvars, n, m = Fv.shape
    
    i = np.arange(js-1, je)    
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
    
    FL = np.moveaxis(FL, 1, axis)
    FR = np.moveaxis(FR, 1, axis)
    return FL, FR

def linear_weno_2d(X, js, je, axis=1):
    """
    5th-order linear WENO reconstruction for 2D array along given axis.
    
    Parameters
    ----------
    X : np.ndarray
        2D array of cell-centered values
    js, je : int
        Interior indices for reconstruction
    axis : int
        Axis along which to reconstruct (1=rows/xi, 0=columns/eta)
    
    Returns
    -------
    Xint : np.ndarray
        2D array of reconstructed midpoints along specifjed axis
    """
    # Move axis to 0 for easjer vectorized indexing
    Xv = np.moveaxis(X, axis, 0)  # shape (n, m)
    Xint = np.zeros_like(Xv)
    n, m = Xv.shape
    i = np.arange(js-2, je+1)
    # factors to blend forward and backward weno differences
    # because the first and last point cannot do backward and
    # and forward with fringe available
    fac_f = np.ones_like(i)*0.5
    fac_b = np.ones_like(i)*0.5
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
    Xint[i, :] = fac_b[:,np.newaxis]*(w0*d0 + w1*d1 + w2*d2)
    d0 = (11*Xp - 7*Xpp + 2*Xppp)/6
    d1 = (2*X0 + 5*Xp - Xpp)/6
    d2 = (-Xm + 5*X0 + 2*Xp)/6
    Xint[i,:] += fac_f[:,np.newaxis]*(w0*d0 + w1*d1 + w2*d2)
    # Move axis back and divide by number of contributions
    Xint = np.moveaxis(Xint, 0, axis)
    return Xint

# -----------------------------
# Flux functions
# -----------------------------
def flux_x(U, gamma=1.4):
    rho = U[0]; u = U[1]/rho; v = U[2]/rho; E = U[3]
    p = (gamma-1)*(E - 0.5*rho*(u**2 + v**2))
    F = np.array([rho*u, rho*u**2 + p, rho*u*v, (E+p)*u])
    return F

def flux_y(U, gamma=1.4):
    rho = U[0]; u = U[1]/rho; v = U[2]/rho; E = U[3]
    p = (gamma-1)*(E - 0.5*rho*(u**2 + v**2))
    G = np.array([rho*v, rho*u*v, rho*v**2 + p, (E+p)*v])
    return G

def max_wave_speed(U, gamma=1.4):
    rho = U[0]; u = U[1]/rho; v = U[2]/rho; E = U[3]
    p = (gamma-1)*(E - 0.5*rho*(u**2 + v**2))
    c = np.sqrt(gamma*p/rho)
    return np.max(np.abs(u) + c), np.max(np.abs(v) + c)

def residual_weno2d(U, x, y, js, je, ks, ke, gamma=1.4):
    """
    Compute 2D WENO5 residual for Euler equations on curvilinear grids.
    Vectorized and interior-only.
    
    Parameters
    ----------
    U : np.ndarray, shape (4, nx, ny)
        Conserved variables [rho, rho*u, rho*v, E], including ghost cells
    x, y : np.ndarray, shape (nx, ny)
        Physical coordinates, including ghost cells
    js, je, ks, ke : int
        Interior index ranges
    gamma : float
        Ratio of specific heats
    
    Returns
    -------
    Res : np.ndarray, shape (4, nx, ny)
        Residuals at interior cells (ghosts remain 0)
    """
    Res = np.zeros_like(U)
    n, m = x.shape
    # --- Reconstruct midpoints ---
    xint_xi  = linear_weno_2d(x, js, je, axis=0)
    yint_xi  = linear_weno_2d(y, js, je, axis=0)
    xint_et = linear_weno_2d(x, ks, ke, axis=1)
    yint_et = linear_weno_2d(y, ks, ke, axis=1)

    # --- Compute metrics at interior points and one
    # --- extra point on either side to average for interfaces
    i = np.arange(js-1, je+1)
    j = np.arange(ks-1, ke+1)
    
    dx_dxi  = xint_xi[i,ks-1:ke+1] - xint_xi[i-1,ks-1:ke+1]
    dy_dxi  = yint_xi[i,ks-1:ke+1] - yint_xi[i-1,ks-1:ke+1]
    dx_det  = xint_et[js-1:je+1,j] - xint_et[js-1:je+1,j-1]
    dy_det  = yint_et[js-1:je+1,j] - yint_et[js-1:je+1,j-1]
    J = dx_dxi*dy_det - dx_det*dy_dxi

    xi_x = dy_det/J
    xi_y = -dx_det/J
    eta_x = -dy_dxi/J
    eta_y = dx_dxi/J

    # --- Loop over conserved variables ---
    F = flux_x(U, gamma)  # x-direction flux
    G = flux_y(U, gamma)  # y-direction flux
    
    # --- Reconstruct interface fluxes and field values ---
    FL_xi, FR_xi = weno5_fd_2d(F, js, je, axis=1)
    GL_xi, GR_xi = weno5_fd_2d(G, js, je, axis=1)
    UL_xi, UR_xi = weno5_fd_2d(U, js,je,axis=1)
    
    FL_eta, FR_eta = weno5_fd_2d(F, ks, ke, axis=2)
    GL_eta, GR_eta = weno5_fd_2d(G, ks, ke, axis=2)
    UL_eta, UR_eta = weno5_fd_2d(U, ks,ke, axis=2)
    
    # --- Max wave speed for Lax-Friedrichs type dissipation ---
    # just arithmetic average now, can change this to Roe average later
    rho_avg = (UL_xi[0][js-1:je,ks:ke]+UR_xi[0][js-1:je,ks:ke])*0.5
    u_avg = (UL_xi[1][js-1:je,ks:ke] + UR_xi[1][js-1:je,ks:ke])*0.5 / rho_avg
    v_avg = (UL_xi[2][js-1:je,ks:ke] + UR_xi[2][js-1:je,ks:ke])*0.5 / rho_avg
    p_avg = (gamma-1)*( (UL_xi[3][js-1:je,ks:ke]+UR_xi[3][js-1:je,ks:ke])*0.5 - 0.5*rho_avg*(u_avg**2 + v_avg**2) )
    c = np.sqrt(gamma*p_avg/rho_avg)
    # these are the metrics at the interface (think of them like a face normal)
    xi_x_avg = (xi_x[:-1,1:-1]+xi_x[1:,1:-1])*0.5
    xi_y_avg = (xi_y[:-1,1:-1]+xi_y[1:,1:-1])*0.5
    V_xi = u_avg*xi_x_avg + v_avg*xi_y_avg
    metric_mag = np.sqrt(xi_x_avg**2 + xi_y_avg**2)
    # wave-speed in contravariant coordinate for dissipation scaling
    # divide by xi_x_avg assuming xi is the dominant direction to physical x coord
    alpha_xi = (np.abs(V_xi) + c*metric_mag)/xi_x_avg    

    # add dissipation term only to xi flux (again assuming xi is dominant along x)
    F_xi = 0.5*(FL_xi[:,js-1:je,ks:ke] + FR_xi[:,js-1:je,ks:ke]) \
        - 0.5*alpha_xi*(UR_xi[:,js-1:je,ks:ke] - UL_xi[:,js-1:je,ks:ke])

    G_xi = 0.5*(GL_xi[:,js-1:je,ks:ke] + GR_xi[:,js-1:je,ks:ke]) 

    # wave-speed in contravariant coordinate for dissipation scaling
    # divide by xi_x_avg assuming xi is the dominant direction to physical x coord
    rho_avg = (UL_eta[0][js:je,ks-1:ke]+UR_eta[0][js:je,ks-1:ke])*0.5
    u_avg = (UL_eta[1][js:je,ks-1:ke] + UR_eta[1][js:je,ks-1:ke])*0.5 / rho_avg
    v_avg = (UL_eta[2][js:je,ks-1:ke] + UR_eta[2][js:je,ks-1:ke])*0.5 / rho_avg
    p_avg = (gamma-1)*( (UL_eta[3][js:je,ks-1:ke]+UR_eta[3][js:je,ks-1:ke])*0.5 - 0.5*rho_avg*(u_avg**2 + v_avg**2) )
    c = np.sqrt(gamma*p_avg/rho_avg)
    eta_x_avg = (eta_x[1:-1,:-1]+eta_x[1:-1,1:])*0.5
    eta_y_avg = (eta_y[1:-1,:-1]+eta_y[1:-1,1:])*0.5
    V_eta = u_avg*eta_x_avg + v_avg*eta_y_avg
    metric_mag = np.sqrt(eta_x_avg**2 + eta_y_avg**2)
    # wave-speed in contravariant coordinate for dissipation scaling
    # divide by eta_y_avg assuming eta is the dominant direction to physical y coord
    alpha_eta = (np.abs(V_eta) + c*metric_mag)/eta_y_avg    

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
def rk3_tvd(U0, x, y, dt, js, je, ks, ke, gamma=1.4):
    residual = residual_weno2d
    R0 = residual(U0, x, y, js, je, ks, ke, gamma)
    U1 = U0 + dt*R0
    R1 = residual(U1, x, y, js, je, ks, ke, gamma)
    U2 = 0.75*U0 + 0.25*(U1 + dt*R1)
    R2 = residual(U2, x, y, js, je, ks, ke, gamma)
    U3 = (U0 + 2*(U2 + dt*R2)) / 3.0
    return U3, np.linalg.norm(U3-U0)

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
    x = np.linspace(-Lx/2, Lx/2, nx)
    y = np.linspace(-Ly/2, Ly/2, ny)
    XX, YY = np.meshgrid(x, y, indexing='ij')
    # make a wavy grid
    X = XX + 0.1*np.sin(YY)
    Y = YY + 0.1*np.sin(XX)
    beta = 1.0
        
    r2 = (X-x0)**2 + (Y-y0)**2
    u = u0 - beta/(2*np.pi)*(Y-y0)*np.exp(0.5*(1-r2))
    v = v0 + beta/(2*np.pi)*(X-x0)*np.exp(0.5*(1-r2))
    T = 1.0 - ((gamma-1)*beta**2)/(8*gamma*np.pi**2)*np.exp(1-r2)
    rho = T**(1/(gamma-1))
    p = rho**(gamma)
    E = p/(gamma-1) + 0.5*rho*(u**2 + v**2)
    
    U = np.zeros((4,nx,ny))
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
    X = XX[3:n-3,3:m-3]
    Y = YY[3:n-3,3:m-3]
    rho = U[0][3:n-3,3:m-3]
    plt.figure(figsize=(6,5))
    cp = plt.contourf(X, Y, rho, levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.plot(X,Y,'k--',linewidth=0.1)
    plt.plot(X.T,Y.T,'k--',linewidth=0.1)
    plt.axis('equal')
    plt.tight_layout()

def plot_density_line(X, U, Uf, title='Density'):
    rho = U[0]
    rhof = Uf[0]
    plt.figure(figsize=(6,5))
    n,m = X.shape
    print(n,m)
    print(U.shape)
    plt.plot(X[:,m//2+1],rho[:,m//2+1],'r.-')
    plt.plot(X[:,m//2+1],rhof[:,m//2+1],'bo-')
    plt.show()
    
# -----------------------------
# Main routine
# -----------------------------
def main():
    # Grid parameters
    nx, ny = 41, 41        # grid points
    Lx, Ly = 10.0, 10.0    # physical domain
    CFL = 5.0              # CFL number
    #t_final = 2         # final time
    gamma = 1.4
    u0=0.5
    # Initialize
    U, X, Y = init_isentropic_vortex(nx, ny, Lx=Lx, Ly=Ly, u0=u0, gamma=gamma)
    # find distance to travel to return back to initial location periodically
    # interior domain size + dx
    t_final = (X[nx-4,ny//2+1]-X[3,ny//2+1])/u0 + (X[1,ny//2+1]-X[0,ny//2+1])/u0
    print("t_final=",t_final)

    apply_periodic(U)
    
    # Plot initial density
    plot_density(X, Y, U, title='Initial Density')
    
    # Time stepping loop
    t = 0.0
    dt = 0.1  # initial time step; can compute CFL-based later
    js, je = 3, nx-3
    ks, ke = 3, ny-3
    #t_final = 0.1
    
    while t < t_final:
        # Apply periodic BCs
        # Compute max wave speeds for CFL-based dt
        t += dt
        alpha_x, alpha_y = max_wave_speed(U[:,js:je,ks:ke], gamma)
        dt_cfl = CFL * min( (X[1,0]-X[0,0])/np.max(alpha_x),
                            (Y[0,1]-Y[0,0])/np.max(alpha_y) )
        dt = min(dt, t_final-t, dt_cfl)        
        # Advance one time step with RK3-TVD
        U, dUnorm = rk3_tvd(U, X, Y, dt, js, je, ks, ke, gamma)
        print(f"time :{t:0.3f} {dUnorm:0.3f}")
    
    # Plot final density
    plot_density(X, Y, U, title=f'Density at t={t_final:.3f}')
    plt.show()
# -----------------------------
# Main routine for checking error
# ----------------------------- 
def errorcheck():
    max_p = 8
    N0 = 10
    for p in range(1,max_p+1): 
        ny =  int(N0 * 1.5**(p-1))+1
        nx =  2 * int(N0 * 1.5**(p-1))+1
        Lx, Ly = 20.0, 10.0    # physical domain
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
    
        while t < t_final:
            # Compute max wave speeds for CFL-based dt
            alpha_x, alpha_y = max_wave_speed(U[:,js:je,ks:ke], gamma)
            dt_cfl = CFL * min( (X[1,0]-X[0,0])/np.max(alpha_x),
                                (Y[0,1]-Y[0,0])/np.max(alpha_y) )
            dt = min(dt, t_final-t, dt_cfl)
        
            # Advance one time step with RK3-TVD
            U, dUnorm = rk3_tvd(U, X, Y, dt, js, je, ks, ke, gamma)
            t += dt
            print(f"time :{t:0.3f} {dUnorm:0.3f}")
    
            # Plot final density
            #plot_density(X, Y, U, title=f'Density at t={t_final:.3f}')
        err = np.sqrt(np.mean(np.square(U[0,js:je,ks:ke]-Uf[0,js:je,ks:ke])))
        # Plot final density
        plot_density_line(X, U, Uf, title=f'Density at t={t_final:.3f}')
        if p > 1:
            print(f"dx: {dx:0.5f} Convergence order: {np.log(errprev/err)/np.log(1.5):.3f}, Error: {err:.3e}")
        else:
            print(f"dx: {dx:0.5f} Error: {err:.3e}")
        errprev=err
# -----------------------------
# Run the solver
# -----------------------------
if __name__ == "__main__":
    #main()
    errorcheck()
    
