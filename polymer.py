#-*- coding: utf-8 -*-
"""
MultiBlock
=========

:copyright: (c) 2015 by Jiuzhou Tang
:license: BSD, see LICENSE for more details.

"""

import numpy as np
from scipy import optimize
import scipy.io
import matplotlib.pyplot as pl
import time

def polymer_scft(D):
    Nx=100
    Ns=200
    qf_y=np.zeros((Ns+1,Nx,Nx))
    qf=np.zeros((Ns+1,Nx))
    qb=np.zeros((Ns+1,Nx))
    rhof=np.zeros((Ns+1,Nx,Nx))
    rho=np.zeros((Nx,Nx))
    gy=np.zeros((Nx))
    ds=1.0/Ns
    dx=D/Nx
    chemw=np.zeros((Nx))
    phi=np.zeros((Nx))
    exp_w=np.zeros((Nx))
    exp_k2=np.zeros((Nx))
    d_kx=2*np.pi/D
    half_kx=Nx/2
    kx=np.arange(Nx,dtype=float)
    kx[:half_kx+1]=kx[:half_kx+1]*d_kx
    kx[half_kx+1:]=d_kx*(Nx-kx[half_kx+1:])
    for i in np.arange(Nx):
    #for i in np.arange(Nx/2):
        chemw[i]=3.0*np.pi**2*(i*dx)**2
        #chemw[Nx-i-1]=chemw[i]
    for i in np.arange(Nx):
        exp_w[i]=np.exp(chemw[i]*(-0.5*ds))
    #pl.plot(exp_w[:],'r')
    pl.plot(-1.0*chemw[:],'r')
    pl.show()
    # initial condition for propagator qf[z,z0,s] in ref[J.C.P vol 117,No. 5]
    local_data=np.zeros((Nx),dtype=complex)
    for x in np.arange(Nx) :
        k2=kx[x]**2
        exp_k2[x]=np.exp(-1.0*ds*k2)

    qf[0,:]=1.0
    for s in np.arange(1,Ns+1):
        local_data[:]=qf[s-1,:]*exp_w[:]+0.0j
        local_data=np.fft.fftn(local_data)
        local_data[:]=local_data[:]*exp_k2[:]
        local_data=np.fft.ifftn(local_data)
        qf[s,:]=local_data[:].real*exp_w[:]
    qb[Ns,:]=0.0
    
    for z0 in np.arange(Nx) :
        qf_y[0,z0,:]=0.0
        qf_y[0,z0,z0]=1.0*Nx
        for s in np.arange(1,Ns+1):
            local_data[:]=qf_y[s-1,z0,:]*exp_w[:]+0.0j
            local_data=np.fft.fftn(local_data)
            local_data[:]=local_data[:]*exp_k2[:]
            local_data=np.fft.ifftn(local_data)
            qf_y[s,z0,:]=local_data[:].real*exp_w[:]


    qb[Ns,:]=0.0
    for x in np.arange(Nx):
        qb[Ns,x]=np.exp(-2000.0*(x*1.0/Nx)**2)*Nx
    #pl.plot(qb[Ns,:],'r^')
    #pl.show()
    
    for x in np.arange(Nx) :
          k2=kx[x]**2
          exp_k2[x]=np.exp(-1.0*ds*k2)
    for s in np.arange(0,Ns)[::-1]:
        local_data[:]=qb[s+1,:]*exp_w[:]+0.0j
        local_data=np.fft.fftn(local_data)
        local_data[:]=local_data[:]*exp_k2[:]
        local_data=np.fft.ifftn(local_data)
        qb[s,:]=local_data[:].real*exp_w[:]


    for x in np.arange(Nx):
        phi[x]=0.0
        for s in np.arange(0,Ns+1):
            phi[x]=phi[x]+qf[s,x]*qb[s,x]
        #print "phi[x]=",phi[x]    
    #pl.plot(phi[:],'r')
    #pl.plot(qf[:,10],'r')
    #pl.imshow(qb+qf)
    #pl.show()
    #for z0 in np.arange(Nx):
    #    for z in np.arange(Nx):
    #        for s in np.arange(Ns+1):
    #            rhof[s,z0,z]=qf_y[s,z0,z]*qf_y[Ns-s,z,0]/qf_y[Ns,z0,0]
    #pl.plot(qf_y[0,50,:])
    pl.plot(qf_y[30,90,:])
    #pl.plot(qf_y[1,50,:])
    #pl.plot(qf_y[3,50,:])
    #pl.plot(qf_y[5,50,:])
    #pl.plot(qf_y[10,50,:])
    #pl.plot(qf_y[20,50,:])
    #pl.plot(qf_y[100,50,:])
    #pl.plot(rhof[5,50,:])
    #pl.plot(rhof[10,50,:])
    #pl.plot(rhof[15,50,:])
    #pl.plot(rhof[20,50,:])
    #pl.plot(rhof[25,50,:])
    #pl.plot(rhof[30,50,:])
    #pl.plot(rhof[35,50,:])
    #pl.plot(rhof[40,50,:])
    pl.show()
    return




def crank_nicolson(Nx,Lx,Nt,dt,chemw,q):
    """
    	This program solves the 1D modified diffusion equation
    		q_t = q_xx-w*q
     
    	The program solves the heat equation using a finite difference method
    	where we use a center difference method in space and Crank-Nicolson 
    	in time.
    """
    
    dx = Lx/Nx
    u = np.transpose(np.mat(q[0,:]))
    I = sparse.identity(Nx+1)
    for i in range(1,Nt+1,1):
        # Second-Derivative Matrix
        data = np.ones((3, Nx+1))
        data[1] = -2*data[1]
        data[1,:]=data[1,:]-chemw[:]*(dx**2)
        # Reflective boundary 
        data[2,1]=2*data[2,1]
        data[0,Nx-1]=2*data[0,Nx-1]
        diags = [-1,0,1]
        D2 = sparse.spdiags(data,diags,Nx+1,Nx+1)/(dx**2)

        A = (I -dt/2*D2)
        b = ( I + dt/2*D2 )*u
        
        u = np.transpose(np.mat( sparse.linalg.spsolve( A,  b ) ))
        q[i,:,0,0]=np.reshape(u[:,0],(Nx+1))





if __name__ == '__main__':
    res=[]
    for size in [5.0]:
        res.append((size,polymer_scft(size)))
    print res
                               
