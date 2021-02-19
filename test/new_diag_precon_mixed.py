import asQ
import firedrake as fd
import numpy as np

import petsc4py.PETSc as PETSc
PETSc.Sys.popErrorHandler() 

#print("Werner's routine")

#Test PCDIAGFFT by using it
#within the relaxation method
#using the heat equation as an example
#we compare one iteration using just the diag PC
#with the direct solver

mesh = fd.UnitSquareMesh(20,20)
V = fd.FunctionSpace(mesh, "CG", 1)

x, y = fd.SpatialCoordinate(mesh)
u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
dt = 0.01
theta = 0.5
alpha = 0.01
M = 4

diagfft_options = {
    'ksp_type':'preonly',
    'pc_type':'lu',
    'pc_factor_mat_solver_type':'mumps',
    'mat_type':'aij'}

solver_parameters = {
    'snes_type':'ksponly',
    'mat_type':'matfree',
    'ksp_type':'preonly',
    'ksp_rtol':1.0e-10,
    'ksp_converged_reason':None,
    'pc_type':'python',
    'pc_python_type':'asQ.DiagFFTPC',
    'diagfft':diagfft_options}

def form_function(u, v):
    return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

def form_mass(u, v):
    return u*v*fd.dx

PD = asQ.paradiag(form_function=form_function,
                        form_mass=form_mass, W=V,
                        w0=u0, dt=dt, theta=theta,
                        alpha=alpha, M=M,
                        solver_parameters=solver_parameters,
                        circ="outside", tol=1.0e-12,
                        maxits=1)
PD.solve(verbose=True)
solver_parameters = {'ksp_type':'preonly', 'pc_type':'lu',
                        'pc_factor_mat_solver_type':'mumps',
                        'mat_type':'aij'}
PDe = asQ.paradiag(form_function=form_function,
                        form_mass=form_mass, W=V,
                        w0=u0, dt=dt, theta=theta,
                        alpha=alpha, M=M,
                        solver_parameters=solver_parameters,
                        circ="outside", tol=1.0e-12,
                        maxits=1)
PDe.solve(verbose=True)
unD = fd.Function(V, name='diag')
un = fd.Function(V, name='full')
err = fd.Function(V, name='error')
unD.assign(u0)
un.assign(u0)
for i in range(M):
    walls = PD.w_all.split()[i]
    wallsE = PDe.w_all.split()[i]
    unD.assign(walls)
    un.assign(wallsE)
    err.assign(un-unD)
    assert(fd.norm(err) < 1.0e-13)

