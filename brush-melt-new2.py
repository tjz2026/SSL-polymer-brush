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
from scipy.integrate import simps
import time
import scipy.optimize # nonlinear solver
from scipy import interpolate

def PolymerBrush_SSL(D):
    global xi # coarse grid on x
    global Nx
    global dx_c
    global delta
    delta=1.0
    Nx=200  # the grid number for x
    Ntheta=200 # the grid number for theta
    dx_c=1.0/(Nx+1.0) # dx for coarse grid
    dx_t=0.5*np.pi/(Ntheta+1.0) # dx_t on theta grid
    xi=np.zeros(Nx)
    x_t=np.zeros(Ntheta)
    y_t=np.zeros(Ntheta)
    g_e_t=np.zeros(Ntheta) # evenly spaced on theta 
    f_t=np.zeros(Ntheta)  # evenly spaced on theta
    for x in np.arange(Nx):
        xi[x]=dx_c*(x+1)
    for x in np.arange(Ntheta):
        x_t[x]=dx_t*(x+1)
        y_t[x]=np.sin(x_t[x])  
    for i in np.arange(0,Ntheta):
        f_t[i]=np.sin(x_t[i])
        g_e_t[i]=np.sqrt(1.0-np.sin(x_t[i])**2)*f_t[i]
    pl.plot(f_t[:],'r^')
    pl.show()
    sum_a1=simps(f_t[0:Nx],dx=dx_t)
    print "g_e sum=",sum_a1

    global Exy
    global Gxy_x_t
    global Gxy_t
    Exy=np.zeros((Ntheta,Ntheta))
    Gxy_x_t=np.zeros((Ntheta,Ntheta)) 
    Gxy_t=np.zeros((Ntheta,Ntheta)) 
    eps=0.0
    xt_t=np.zeros(Ntheta)
    yt_t=np.zeros(Ntheta)
    for i in np.arange(Ntheta): # this is y: sint
        for t in np.arange(Ntheta):
            yt_t[t]=y_t[i]*y_t[t] # x=sint*sint'  
        #for j in np.arange(i+1): # warning, this is wrong, the j index can be as large as Ntheta
        for j in np.arange(Ntheta): # this is x, x is sint*sint' # so, it is actually G(t,t')
            Gxy_x_t[i,j]=np.sqrt(y_t[i]**2-yt_t[j]**2)*(2.0/np.pi)/np.sqrt(y_t[i]**2-yt_t[j]**2+eps)
    for i in np.arange(1,Ntheta): # this is y: the first point missing
        for j in np.arange(i): # this is x, x is sint*sint' # so, it is actually G(t,t')
            Gxy_t[j,i]=np.sqrt(y_t[i]**2-y_t[j]**2)*(2.0/np.pi)/np.sqrt(y_t[i]**2-y_t[j]**2+eps)
    # test the equal length condition
    gama_t=np.zeros(Ntheta)
    for i in np.arange(1,Ntheta): # this is t, (y)
        gama_t[i]=simps(Gxy_x_t[:,i],dx=dx_t)
        #d_factor=dx_t*(Ntheta*1.0/i) 
        #gama_t[i]=simps(Gxy_t[0:i,i],dx=d_factor)

    pl.ylim(0.0, 1.0)
    pl.plot(gama_t[:],'y^',linewidth=2.5)
    pl.legend(['gama_t'])
    pl.show()

    ## doing the integral of local density:
    rho_t=np.zeros(Ntheta)
    int_y=np.zeros(Ntheta)
    global root_sin_y_inv
    global root_sin_y
    root_sin_y_inv=np.zeros((Ntheta,Ntheta))
    root_sin_y=np.zeros((Ntheta,Ntheta))
    for i in np.arange(Ntheta-1): #x  the last point is deleted
        for j in np.arange(i+1,Ntheta):  # y :sint, note that y>x, first point deleted
            if i==j: print "i=j",i,j 
            root_sin_y_inv[i,j]=1.0/np.sqrt(y_t[j]**2-y_t[i]**2)  
            root_sin_y[j,i]=np.sqrt(y_t[j]**2-y_t[i]**2)  
    # this integral of density has two sigularity near each boundaries, needs to be considered
    #carefully.
    for i in np.arange(Ntheta-1): # the last point of x is deleted
        int_y[i+1:Ntheta]=f_t[i+1:Ntheta]*Gxy_t[i,i+1:Ntheta]*root_sin_y_inv[i,i+1:Ntheta]
        int_y[i+1:Ntheta]=int_y[i+1:Ntheta]-(2.0/np.pi)*y_t[i+1:Ntheta]*root_sin_y_inv[i,i+1:Ntheta]
        rho_t[i]=rho_t[i]+simps(int_y[i+1:Ntheta],dx=dx_t)+1.0
    rho_t[Ntheta-1]=1.0 
    pl.plot(rho_t[:],'r^')
    pl.show()

    # starting to solve the 3*Nx nonlinear equations
    X0=np.zeros(3*Ntheta) # intial guess for g(y), phi_x(x) and lambda(y)
    global X,F
    X=np.zeros(3*Ntheta) # the solution 
    F=np.zeros(3*Ntheta) # The functions
    for i in np.arange(Ntheta):
        X0[i]=f_t[i] # f(y)
        X0[i+Nx]=-(3.0/16.0)*delta**2*((np.pi*y_t[i])**2)
        X0[i+2*Nx]=(3.0/16.0)*delta**2*(np.pi**2)*(y_t[i]**3)/(np.sqrt(1.0-y_t[i]**2))

    F[:]=0.0  
    FUNC(X0)
    print "bf F",F[0:300]
    X = scipy.optimize.newton_krylov(FUNC, X0, verbose=True,f_tol=1e-3)
    print "X",X[0:100]
    print "F",F[0:300]
    pl.plot(X[0:300],'r')
    pl.show()

def FUNC(X):
    ## doing the integral of local density:
    # this integral of density has two sigularity near each boundaries, needs to be considered
    #carefully.
    # First, build the E(x,y) function.
    Gxy_build(X)
    Gxy_to_Exy( 
    F[:]=0.0
    int_y=np.zeros(Ntheta)
    phi_t_intp=np.zeros(Ntheta)
    for i in np.arange(Ntheta-1): # the last point of x is deleted
        int_y[i+1:Ntheta]=f_t[i+1:Ntheta]*Gxy_t[i,i+1:Ntheta]*root_sin_y[i,i+1:Ntheta]
        int_y[i+1:Ntheta]=int_y[i+1:Ntheta]-(2.0/np.pi)*y_t[i+1:Ntheta]*root_sin_y_inv[i,i+1:Ntheta]
        F[i]=simps(int_y[i+1:Ntheta],dx=dx_t)
    F[Ntheta-1]=0.0 
    # calculate the F2: equal length condition
    for i in np.arange(Ntheta): # this is t, (y)
        F[i+Ntheta]=simps(Gxy_x_t[i,:],dx=dx_t)-1.0
    #for i in np.arange(Nx): # this is y
    #    for j in np.arange(i+1): # this is x, x is faster
    #        F[i+Nx]=F[i+Nx]+simps(1.0/Exxy[i,j,:],dx=dx_f)
    #    F[i+Nx]=F[i+Nx]-1.0 
    ## calculate the F3:           
    for i in np.arange(Ntheta): # this is y
        phi_t_intp=interpolate_phi(X[Ntheta:2*Ntheta],i) # interpolate the phi(x) on the scaled grid   
        F[i+2*Nx]=simps(Gxy_x_t[i,:]*phi_t_intp[:],dx=dx_t)  
        F[i+2*Nx]=(3.0/4.0)*delta**2*simps(root_sin_y_scale[i,:]*Gxy_t[i,:],dx=dx_t)  

    #pl.plot(F[:],'r^')
    #pl.show()
    return F

def Exy_build(X):
    global Exy
    global Exxy
    global Eyxy
    eps=1.0e-10
    for i in np.arange(Nx): # this is y
        for j in np.arange(i+1): # this is x, x is faster
            Exy[i,j]=(4.0/3.0)*delta**(-2)*(X[j+Nx]+X[i+2*Nx]/X[i])
            if Exy[i,j]<0.0 :
               Exy[i,j]=0.0
            Exy[i,j]=np.sqrt(Exy[i,j]+eps)  


    # the first four grid are replaced by anlytical solution
    error_sum=0.0
    for i in np.arange(0,10): # this is y
        for j in np.arange(i+1): # this is x, x is faster
            for k in np.arange(Nm):
                Exxy[i,j,k]=0.5*np.pi*np.sqrt(xi[i]**2-xij[j,k]**2+eps)
                #error_sum=error_sum+np.abs(Exxy[i,j,k]-Exxy0[i,j,k])
    for i in np.arange(10,Nx): # this is y
        N_intv=i/10
        for ii in np.arange(N_intv):
            x_s=i-10*(ii+1)+1
            x_e=i-10*ii
            intp=np.polyfit(xi[x_s:x_e+1], Exy[i,x_s:x_e+1],6)
            p10 = np.poly1d(intp)
            for j in np.arange(x_s,x_e+1,1): # this is x, x is faster
                Exxy[i,j,:]=p10(xij[j,:])
    #for i in np.arange(10,Nx): # this is y
    #    for j in np.arange(i+1): # this is x, x is faster
    #        for k in np.arange(Nm):
    #            error_sum=error_sum+np.abs(Exxy[i,j,k]-Exxy0[i,j,k])
    #print "now error=",error_sum
    #pl.imshow(Exxy[0,:,:])
    #pl.show()
    #for i in np.arange(0,Nx-10): # this is x
    #    intp=np.polyfit(xi[i:i+10+1], Exy[i:i+10+1,i],8)
    #    p10 = np.poly1d(intp)
    #    for j in np.arange(i,Nx): # this is y, y is faster
    #        Eyxy[i,j,:]=p10(xji[j,:])
    #for i in np.arange(Nx-10,Nx): # this is x
    #    for j in np.arange(i,Nx): # this is y
    #        for k in np.arange(Nm):
    #            Eyxy[i,j,k]=0.5*np.pi*np.sqrt(xji[j,k]**2-xi[i]**2+eps)
    return 
    



if __name__ == '__main__':
    res=[]
    #for size in [2.23]:
    for size in [6.122]:
        res.append((size,PolymerBrush_SSL(size)))
    print res

