import firedrake as fd
import numpy as np
from scipy.fft import fft, ifft

mesh = fd.PeriodicUnitSquareMesh(20,20)
V = fd.VectorFunctionSpace(mesh, "BDM", 1)
Q = fd.VectorFunctionSpace(mesh, "DG", 0)
W = V * Q

x, y = fd.SpatialCoordinate(mesh)
w0 = fd.Function(W)
u0, p0 = w0.split()
p0.interpolate(fd.as_vector([fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2), 0]))
dt = 0.01
theta = 0.5
alpha = 0.001
M = 4
solver_parameters = {'ksp_type':'preonly', 'pc_type':'lu',
                        'pc_factor_mat_solver_type':'mumps',
                        'mat_type':'aij'}

# Gamma coefficients
Nt = M
exponents = np.arange(Nt)/Nt
alphav = 0.01
Gam = alphav**exponents
        
# Di coefficients
thetav = 0.5
Dt = 0.01
C1col = np.zeros(Nt)
C2col = np.zeros(Nt)
C1col[:2] = np.array([1, -1])/Dt
C2col[:2] = np.array([thetav, 1-thetav])
D1 = np.sqrt(Nt)*fft(Gam*C1col)
D2 = np.sqrt(Nt)*fft(Gam*C2col)

u, p = fd.TrialFunctions(W)
v, q = fd.TestFunctions(W)

def form_function(uu, up, vu, vp):
    return (fd.div(vu)*up - fd.div(uu)*vp)*fd.dx

def form_mass(uu, up, vu, vp):
    return (fd.inner(uu, vu) + up*vp)*fd.dx

un = fd.Function(W)

for i in range(M):
    D1i = fd.Constant(np.imag(D1[i]))
    D1r = fd.Constant(np.real(D1[i]))
    D2i = fd.Constant(np.imag(D2[i]))
    D2r = fd.Constant(np.real(D2[i]))

    usr = u[0,:], p[0]
    usi = u[1,:], p[1]
    vsr = v[0,:], q[0]
    vsi = v[1,:], q[1]

    L = (
        D1r*form_mass(*usr, *vsr)
        - D1i*form_mass(*usi, *vsr)
        + D2r*form_function(*usr, *vsr)
        - D2i*form_function(*usi, *vsr)
        + D1r*form_mass(*usi, *vsi)
        + D1i*form_mass(*usr, *vsi)
        + D2r*form_function(*usi, *vsi)
        + D2i*form_function(*usr, *vsi)
    )

    F = fd.inner(q, p0)*fd.dx
    
    diagfft_options = {
        'ksp_type':'gmres',
        'ksp_monitor':None,
        'pc_type':'lu',
        'pc_factor_mat_solver_type':'mumps',
        'mat_type':'aij'}

    fd.solve(L==F, un, solver_parameters=diagfft_options)


