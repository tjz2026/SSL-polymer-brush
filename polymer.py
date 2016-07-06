#-*- coding: utf-8 -*-
"""
MultiBlock
=========

:copyright: (c) 2016 by Jiuzhou Tang
:license: BSD, see LICENSE for more details.

"""

import numpy as np
from scipy import optimize
from scipy import fftpack
import scipy.io
import matplotlib.pyplot as pl
from numpy.linalg import inv
import scipy as sc
import scipy.sparse as sparse
import scipy.sparse.linalg

import time

def polymer_scft(D):
    Nx=400
    Ns=300
    qf_y=np.zeros((Ns+1,Nx,Nx))
    qf=np.zeros((Ns+1,Nx))
    qfr=np.zeros((Ns+1,Nx+1))
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
    exp_k2_dct=np.zeros((Nx))
    d_kx=2*np.pi/D
    half_kx=Nx/2
    kx=np.arange(Nx,dtype=float)
    kx[:half_kx+1]=kx[:half_kx+1]*d_kx
    kx[half_kx+1:]=d_kx*(Nx-kx[half_kx+1:])
    #for i in np.arange(Nx):
        #chemw[i]=3.0*np.pi**2*(i*dx)**2
        #chemw[Nx-i-1]=chemw[i]
    for i in np.arange(Nx):
        exp_w[i]=np.exp(chemw[i]*(-0.5*ds))
    #pl.plot(exp_w[:],'r')
    # initial condition for propagator qf[z,z0,s] in ref[J.C.P vol 117,No. 5]
    local_data=np.zeros((Nx),dtype=complex)
    local_data_rin=np.zeros((Nx))
    local_data_rout=np.zeros((Nx))
    for x in np.arange(Nx) :
        k2=kx[x]**2
        k2_dct=(np.pi/D)*x*1.0
        k2_dct=k2_dct**2
        exp_k2[x]=np.exp(-1.0*ds*k2)
        exp_k2_dct[x]=np.exp(-1.0*ds*k2_dct)

    qf[0,:]=1.0
    for x in np.arange(Nx):
        qf[0,x]=np.exp(-2000.0*((x-Nx/2)*1.0/Nx)**2)*Nx
        chemw[x]=-30*np.cos(np.pi*x*2.0/Nx)
        exp_w[x]=np.exp(chemw[x]*(-0.5*ds))
    pl.plot(exp_w[:],'r')
    pl.show()
    for s in np.arange(1,Ns+1):
        #local_data[:]=qf[s-1,:]*exp_w[:]+0.0j
        #local_data=np.fft.fftn(local_data)
        #local_data[:]=local_data[:]*exp_k2[:]
        #local_data=np.fft.ifftn(local_data)
        #qf[s,:]=local_data[:].real*exp_w[:]
        local_data_rin[:]=qf[s-1,:]*exp_w[:]
        local_data_rin=fftpack.dct(local_data_rin, type=2, norm='ortho')
        local_data_rin[:]=local_data_rin[:]*exp_k2_dct[:]
        local_data_rout=fftpack.idct(local_data_rin, type=2, norm='ortho')
        qf[s,:]=local_data_rout[:]*exp_w[:]
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
    #pl.plot(qf_y[30,90,:])
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
    #pl.show()
    #qfr[0,90]=1.0*Nx 
    for x in np.arange(Nx):
        qfr[0,x]=np.exp(-2000.0*((x-Nx/2)*1.0/Nx)**2)*Nx
        chemw[x]=-30*np.cos(np.pi*x*2.0/Nx)
    crank_nicolson(Nx,D,Ns,ds,chemw,qfr)
    #pl.plot(qfr[:,50])
    #pl.plot(qfr[1,:],'y')
    #pl.plot(qfr[5,:])
    #pl.plot(qfr[10,:])
    pl.plot(qf[Ns/10,:],'b')
    pl.plot(qfr[Ns/10,:],'r^')
    #pl.plot(chemw,'b^')
    #pl.plot(qfr[20,:])
    #pl.plot(qfr[40,:])
    #pl.plot(qf_y[30,90,:])
    Ws=np.append(chemw[:],chemw[0]) 
    #pl.plot(0.1*Ws[:],'bo')
    pl.show()
    print "chemw[0],chemw[99]",chemw[0],chemw[1],chemw[98],chemw[99]
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
        Ws=np.append(chemw[:],chemw[0]) 
        data[1,:]=data[1,:]-Ws[:]*(dx**2)
        # Reflective boundary 
        data[2,1]=2*data[2,1]
        data[0,Nx-1]=2*data[0,Nx-1]
        diags = [-1,0,1]
        D2 = sparse.spdiags(data,diags,Nx+1,Nx+1)/(dx**2)

        A = (I -dt/2*D2)
        b = ( I + dt/2*D2 )*u
        
        u = np.transpose(np.mat( sparse.linalg.spsolve( A,  b ) ))
        q[i,:]=np.reshape(u[:,0],(Nx+1))





if __name__ == '__main__':
    res=[]
    for size in [5.0]:
        res.append((size,polymer_scft(size)))
    print res
                               
