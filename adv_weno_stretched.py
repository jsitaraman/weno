import numpy as np
import matplotlib.pyplot as plt
from distrib import distrib
#
# solve linear advection u_t  + a u_x = 0
# on a stretch Cartesian grid with periodic BCs
#
def weno5_fd(f, is_, ie):
    """
    5th-order fd midpoint reconstruction for left and right states
    vectorized

    parameters
    ----------
    f : np.ndarray
        cell-centered values
    is_ : int
        start index (python indexing)
    ie : int
        end index
    returns
    -------
    fintL, fintR : left and right reconstructed fluxes at midpoints
    """
    mdim = len(f)
    fintL = np.zeros(mdim)
    fintR = np.zeros(mdim)
    eps = 1e-6
    # --- interior fd-weno derivative ---
    i=np.arange(is_-1,ie)
    fmm, fm, f0, fp, fpp, fppp = f[i-2], f[i-1], f[i], f[i+1], f[i+2],f[i+3]
    
    # candidate 3rd-order reconstructions for f+
    d0 = (2*fmm - 7*fm + 11*f0)/6.0
    d1 = (-fm + 5*f0 + 2*fp)/6.0
    d2 = (2*f0 + 5*fp - fpp)/6.0
   
    # smoothness indicators
    beta0 = 13/12*(fmm - 2*fm + f0)**2 + 1/4*(fmm - 4*fm + 3*f0)**2
    beta1 = 13/12*(fm - 2*f0 + fp)**2 + 1/4*(fm - fp)**2
    beta2 = 13/12*(f0 - 2*fp + fpp)**2 + 1/4*(3*f0 - 4*fp + fpp)**2
    
    # nonlinear weights
    alpha0 = 0.1 / (eps + beta0)**2
    alpha1 = 0.6 / (eps + beta1)**2
    alpha2 = 0.3 / (eps + beta2)**2
    wsum = alpha0 + alpha1 + alpha2
    w0, w1, w2 = alpha0/wsum, alpha1/wsum, alpha2/wsum
        
    # fd weno reconsruction  (left state)
    fintL[i] = (w0*d0 + w1*d1 + w2*d2)

    # candidate 3rd-order reconstructions for f-
    d0 = (11*fp -7*fpp + 2*fppp)/6.0
    d1 = (2*f0 + 5*fp  - 1*fpp)/6.0
    d2 = (-1*fm+ 5*f0  + 2*fp)/6.0

    beta0 = 13/12*(fp - 2*fpp + fppp)**2 + 1/4*(3*fp - 4*fpp + fppp)**2
    beta1 = 13/12*(f0 - 2*fp + fpp)**2 + 1/4*(f0 - fpp)**2
    beta2 = 13/12*(fm - 2*f0 + fp)**2 + 1/4*(fm - 4*f0 + 3*fp)**2

    # nonlinear weights
    alpha0 = 0.1 / (eps + beta0)**2
    alpha1 = 0.6 / (eps + beta1)**2
    alpha2 = 0.3 / (eps + beta2)**2
    wsum = alpha0 + alpha1 + alpha2
    w0, w1, w2 = alpha0/wsum, alpha1/wsum, alpha2/wsum

    # fd-weno reconstruction (right state)
    fintR[i] = (w0*d0 + w1*d1 + w2*d2)
    return fintL, fintR


def linear_weno(x, is_, ie):
   """
    5th-order fd midpoint reconstruction of coordinates
    vectorized

    parameters
    ----------
    x : np.ndarray
        cell-centered values of coordinates
    is_ : int
        start index (python indexing)
    ie : int
        end index
    returns
    -------
    xint : left and right reconstructed fluxes at midpoints
    """
   i=np.arange(is_-1,ie)
   mdim = len(x)
   xint = np.zeros(mdim)   
   w0, w1, w2 = 0.1,0.6,0.3
   xmm, xm, x0, xp, xpp, xppp = x[i-2], x[i-1], x[i], x[i+1], x[i+2],x[i+3]
   d0 = (2*xmm - 7*xm + 11*x0)/6.0
   d1 = (-xm + 5*x0 + 2*xp)/6.0
   d2 = (2*x0 + 5*xp - xpp)/6.0
   xint[i] = (w0*d0+w1*d1+w2*d2)
   return xint

def avg_2nd(x, is_, ie):
   i=np.arange(is_-1,ie)
   mdim = len(x)
   xint = np.zeros(mdim)   
   xint[i] = (x[i]+x[i+1])*0.5
   return xint

# ------------------------ Test Harness ------------------------

def Residual(u,x,t):
    N=u.shape[0]-6
    is_= 3
    ie=N+3
    a=1
    flux = a*u
    fintL,fintR = weno5_fd(flux,is_,ie) #f_{j+1/2} [+/-]
    uintL,uintR = weno5_fd(u,is_,ie)    #u_{j+1/2} [+/-]
    xint        = linear_weno(x,is_,ie) #x_{j+1/2} [+/-]
    #xint        = avg_2nd(x,is_,ie) #x_{j+1/2} [+/-]
    fint = (fintL+fintR)*0.5 + abs(a)*(uintL - uintR)*0.5
    #xint = (xintL+xintR)*0.5
    i = np.arange(is_,ie)
    fdot=np.zeros_like(u)
    fdot[i] = (fint[i]-fint[i-1])/(xint[i]-xint[i-1])
    bc(fdot,is_,ie)
    return -fdot

def bc(u,is_,ie):
    umid = u[is_:ie]
    u[0:is_]=umid[-is_:]
    u[-is_:]=umid[:is_]

# Sine wave test with harmonics
max_p = 8
N0 = 16
cfl=0.1
k=100

for p in range(1, max_p+1):
    N = N0 * 2**(p-1)+1
    is_ = 3
    ie = N + 3
    MDIM = N + 6

    x = np.zeros(MDIM)
    u = np.zeros(MDIM)
    dx = 1.0/N
    # initialize stretched grid
    xx=distrib(3,[None, 1,N//2+1,N+1],[None, 0,0.5,1],[None, 10*dx,0.2*dx,10*dx])[1:N+1]
    #print(xx)
    x[is_:ie] = xx
    for i in range(is_,ie):
        u[i] = 0.1*np.exp(-k*(x[i]-0.5)**2)
    bc(x,is_,ie)
    x[:is_] = x[:is_]-1
    x[-is_:] = 1+x[-is_:]
    #print("x=",x)
    bc(u,is_,ie)
    #plt.plot(x,u,'rd-')
    #plt.plot(x[is_:ie],u[is_:ie],'bo-')
    u0=u.copy()
    #
    dt = cfl*dx
    nsteps=int(1.0/dt)
    method='rk4'
    for n in range(nsteps):
        t=n*dt
        if method=='rk3':
          u1 = u +  dt*Residual(u,x,t)
          u2 = 0.75*u + 0.25*u1 + 0.25*dt*Residual(u1,x,t+dt)
          u  = (1/3)*u + (2/3)*u2 + 2/3*dt*Residual(u2,x,t+0.5*dt)
        elif method=='rk4':
            k1=Residual(u,x,t)
            k2=Residual(u + 0.5*dt*k1,x,t+0.5*dt)
            k3=Residual(u + 0.5*dt*k2,x,t+0.5*dt)
            k4=Residual(u + dt*k3,x,t+dt)
            u = u + (dt/6)*(k1+2*k2+2*k3+k4)
        #print(n+1,(n+1)*dt)
    plt.plot(x[is_:ie+1],u[is_:ie+1],'rs-')
    err = np.sqrt(np.mean(np.square(u[is_:ie]-u0[is_:ie])))
    plt.plot(x[is_:ie+1],u0[is_:ie+1],'bo-')
    plt.show()
    if p > 1:
        print(f"dx: {dx:0.5f} Convergence order: {np.log(errprev/err)/np.log(2):.3f}, Error: {err:.3e}")
    else:
        print(f"dx: {dx:0.5f} Error: {err:.3e}")
    errprev = err
