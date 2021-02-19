import asQ
import firedrake as fd
import numpy as np

import petsc4py.PETSc as PETSc 
#PETSc.Sys.popErrorHandler() 

#checks that the all-at-once system is the same as solving
#timesteps sequentially using the mixed wave equation as an
#example by substituting the sequential solution and evaluating
#the residual

mesh = fd.PeriodicUnitSquareMesh(20,20)
V = fd.FunctionSpace(mesh, "BDM", 1)
Q = fd.FunctionSpace(mesh, "DG", 0)
W = V * Q

x, y = fd.SpatialCoordinate(mesh)
w0 = fd.Function(W)
u0, p0 = w0.split()
p0.interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
dt = 0.01
theta = 0.5
alpha = 0.001
M = 4
solver_parameters = {'ksp_type':'preonly', 'pc_type':'lu',
                        'pc_factor_mat_solver_type':'mumps',
                        'mat_type':'aij'}
def form_function(uu, up, vu, vp):
    return (fd.div(vu)*up - fd.div(uu)*vp)*fd.dx

def form_mass(uu, up, vu, vp):
    return (fd.inner(uu, vu) + up*vp)*fd.dx


diagfft_options = {
    'ksp_type':'preonly',
    'pc_type':'lu',
    'pc_factor_mat_solver_type':'mumps',
    'mat_type':'aij'}

solver_parameters_diag = {
    'snes_type':'ksponly',
    'mat_type':'matfree',
    'ksp_type':'preonly',
    'ksp_rtol':1.0e-10,
    'ksp_converged_reason':None,
    'pc_type':'python',
    'pc_python_type':'asQ.DiagFFTPC',
    'diagfft':diagfft_options}


PD = asQ.paradiag(form_function=form_function,
                        form_mass=form_mass, W=W, w0=w0, dt=dt,
                        theta=theta, alpha=alpha, M=M,
                        solver_parameters=solver_parameters_diag,
                        circ="outside", tol=1.0e-12)
PD.solve(verbose=True)
#PD.solve()





#sequential solver
un = fd.Function(W)
unp1 = fd.Function(W)

un.assign(w0)
v = fd.TestFunction(W)

eqn = form_mass(*(fd.split(unp1)), *(fd.split(v)))
eqn -= form_mass(*(fd.split(un)), *(fd.split(v)))
eqn += fd.Constant(dt*(1-theta))*form_function(*(fd.split(un)),
                                                *(fd.split(v)))
eqn += fd.Constant(dt*theta)*form_function(*(fd.split(unp1)),
                                            *(fd.split(v)))

sprob = fd.NonlinearVariationalProblem(eqn, unp1)
solver_parameters = {'ksp_type':'preonly', 'pc_type':'lu',
                        'pc_factor_mat_solver_type':'mumps',
                        'mat_type':'aij'}
ssolver = fd.NonlinearVariationalSolver(sprob, solver_parameters=solver_parameters)
ssolver.solve()

err = fd.Function(W, name="err")
pun = fd.Function(W, name="pun")
puns = pun.split()
for i in range(M):
    ssolver.solve()
    un.assign(unp1)
    walls = PD.w_all.split()[2*i:2*i+2]
    for k in range(2):
        puns[k].assign(walls[k])
    err.assign(un-pun)
    assert(fd.norm(err) < 1.0e-15)
 
