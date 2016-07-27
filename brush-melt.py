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
    global xij # fine grid interpolated on x
    global xji # fine grid interpolated on y
    global Nx
    global Nm
    global dx_f
    global dx_c
    Nx=100
    dx_c=1.0/(Nx+1.0) # dx for coarse grid
    xi=np.zeros(Nx)
    gama=np.zeros(Nx)
    E0y=np.zeros(Nx)
    lambda_y=np.zeros(Nx)
    g_e=np.zeros(Nx)
    for x in np.arange(Nx):
        xi[x]=dx_c*(x+1)
        E0y[x]=xi[x]  
        #print "xi[x]",xi[x]
    Nm=200
    Nf=20 
    dxx=1.0/(Nm+1)
    dx_f=dx_c/(Nm+1.0)
    xij=np.zeros((Nx,Nm)) # x grid interpolate on x
    xji=np.zeros((Nx,Nm)) # y grid interpolate on y
    for i in np.arange(Nx): # x
        for j in np.arange(Nm):
            xij[i,j]=(xi[i]-dx_c)+(j+1)*(dx_f)
    for i in np.arange(Nx): # y
        for j in np.arange(Nm):
            xji[i,j]=(xi[i])+(j+1)*(dx_f) # note the starting and ending point 
    #print "xij[Nx-1]",xij[Nx-1,:]
    #print "xji[Nx-1]",xji[Nx-1,:]
    # initial guess for g(y)
    for i in np.arange(Nx):
        g_e[i]=xi[i]/np.sqrt((1.0-xi[i]**2))
    pl.plot(np.log(g_e[Nx-Nf:200]),'r^')
    pl.show()
    sum_a1=simps(g_e[1:Nx]/E0y[1:Nx],dx=dx_c)
    print "sum_a1=",sum_a1

    # doing the polynomial interpolation for the last ten grid,
    xx_tmp=np.zeros(Nf) # last N_f grids
    yy_tmp=np.zeros(Nf)
    Nx_int_ge=2000 # interpolate grid number, from xi[Nx-10]~1.0!
    ge_intp0=np.zeros(Nx_int_ge)
    x_intp = np.linspace(xi[Nx-Nf], 1.0, Nx_int_ge)
    xx_tmp[:]=xi[Nx-Nf:Nx]  
    yy_tmp[:]=(g_e[Nx-Nf:Nx])  
    #yy_tmp[:]=np.log(g_e[Nx-Nf:Nx])  
    intp=np.polyfit(xx_tmp, yy_tmp, 8)
    p1 = np.poly1d(intp)
    ge_intp0[:]=p1(x_intp)
    sum_g_e=simps(g_e[0:Nx-Nf+1],dx=dx_c)
    sum_g_e=sum_g_e+simps((ge_intp0[:]),dx=dx_c*Nf/(Nx_int_ge))
    #sum_g_e=sum_g_e+simps(np.exp(ge_intp[:]),dx=dx_c*Nf/(Nx_int_ge))
    pl.plot(ge_intp0,'b')
    pl.show()
    print "sum of g_e=",sum_g_e

    global Exy
    Exy=np.zeros((Nx,Nx))
    eps=0.0
    for i in np.arange(Nx): # this is y
        for j in np.arange(i+1): # this is x, x is faster
            if xi[i]**2-xi[j]**2 +eps <0.0:
                print "xi[i]",xi[i],xi[j],i,j   
            Exy[i,j]=0.5*np.pi*np.sqrt(xi[i]**2-xi[j]**2+eps)
    #print "E0y[:,0],",Exy[:,0]/(0.5*np.pi*xi[:])
    global Exxy
    global Eyxy
    Exxy=np.zeros((Nx,Nx,Nm)) # interpolate on x
    Eyxy=np.zeros((Nx,Nx,Nm)) # interpolate on y
    for i in np.arange(Nx): # this is y
        for j in np.arange(i+1): # this is x, x is faster
            for k in np.arange(Nm):
                if xi[i]**2-xij[j,k]**2 +eps <0.0:
                    print "xi[i]",xi[i],xij[j,k]   
                Exxy[i,j,k]=0.5*np.pi*np.sqrt(xi[i]**2-xij[j,k]**2+eps)
    for i in np.arange(Nx): # this is x
        for j in np.arange(i,Nx): # this is y, y is faster, and y is always larger than /or equal to x
            for k in np.arange(Nm):
                Eyxy[i,j,k]=0.5*np.pi*np.sqrt(xji[j,k]**2-xi[i]**2+eps)
    
    # test the equal length condition
    # for yi<=Nx_intpo, we interpolate on all x 
    Nx_intpo=10
    for i in np.arange(Nx_intpo): # this is y
        for j in np.arange(i+1): # this is x, x is faster
            gama[i]=gama[i]+simps(1.0/Exxy[i,j,:],dx=dx_f)
    for i in np.arange(Nx_intpo,Nx): # this is y , for yi>=20, we only interpolate the last ten  grid point.
        intpo_grid=10
        gama[i]=gama[i]+simps(1.0/Exy[i,0:i-intpo_grid+1],dx=dx_c)
        for k in np.arange(intpo_grid+1):
            #gama[i]=gama[i]+simps(1.0/Exxy[i,i-intpo_grid+1+k,:],dx=1.0/(101.0*(Nm+1)))
            gama[i]=gama[i]+simps(1.0/Exxy[i,i-intpo_grid+k,:],dx=dx_f)

    pl.ylim(0.0, 1.0)
    pl.plot(gama[:],'y^',linewidth=2.5)
    pl.legend(['Nx=100, Nm=200'])
    pl.show()
    print "gama[:]",gama[:]
    ## doing the integral of local density:
    phi_total=np.zeros(Nx)
    # this integral of density has two sigularity near each boundaries, needs to be considered
    #carefully.
    ge_intp=np.zeros(Nx*Nm)
    xxx=np.zeros((2,Nm))
    XX=np.zeros(10)
    XX[:]=np.sqrt(1.0-xi[Nx-10:Nx]**2)
    intp=np.polyfit(xi[Nx-10:Nx], g_e[Nx-10:Nx]*XX[:],6)
    p10 = np.poly1d(intp)
    # interpolate g(y)*np.sqrt(1.0-g(y)^2)
    for i in np.arange(Nx/10):
        blk_s=i*10
        blk_e=(i+1)*10
        XX[:]=np.sqrt(1.0-xi[blk_s:blk_e]**2)
        intp=np.polyfit(xi[blk_s:blk_e], g_e[blk_s:blk_e]*XX[:],6)
        p10 = np.poly1d(intp)
        for j in np.arange(blk_s,blk_e,1):
            ge_intp[j*Nm:(j+1)*Nm]=p10(xji[j,0:Nm])
              
    for i in np.arange(0,Nx,1):
        for j in np.arange(i,Nx,1):
            phi_total[i]=phi_total[i]+simps(ge_intp[j*Nm:(j+1)*Nm]/(np.sqrt(1.0-xji[j,:]**2)*Eyxy[i,j,:]),dx=dx_f)
        print "phi=",phi_total[i],i 
 
    pl.plot(phi_total,'r^')
    pl.show()

    # starting to solve the 3*Nx nonlinear equations
    X0=np.zeros(3*Nx) # intial guess for g(y), phi_x(x) and lambda(y)
    X=np.zeros(3*Nx) # the solution 
    F=np.zeros(3*Nx) # The functions
    for i in np.arange(Nx):
        X0[i]=g_e[i]
        X0[i+Nx]=-(3.0/16.0)*delta**2*((np.pi*xi[i])**2)
        X0[i+2*Nx]=(3.0/16.0)*delta**2*(np.pi**2)*(xi[i]**3)/(np.sqrt(1.0-xi[i]**2))
    FUNC(X0)
    #X = scipy.optimize.newton_krylov(FUNC, X0, verbose=True,f_tol=1e-3)


def FUNC(X):
    ## doing the integral of local density:
    # this integral of density has two sigularity near each boundaries, needs to be considered
    #carefully.
    # First, build the E(x,y) function.
    Exy_build(X)

    ge_intp=np.zeros(Nx*Nm)
    x_intp=np.zeros(10)
    F[:]=0.0
    # interpolate g(y)*np.sqrt(1.0-g(y)^2)
    for i in np.arange(Nx/10):
        blk_s=i*10
        blk_e=(i+1)*10
        x_intp[:]=np.sqrt(1.0-xi[blk_s:blk_e]**2)
        intp=np.polyfit(xi[blk_s:blk_e], X[blk_s:blk_e]*x_intp[:],6)
        p10 = np.poly1d(intp)
        for j in np.arange(blk_s,blk_e,1):
            ge_intp[j*Nm:(j+1)*Nm]=p10(xji[j,0:Nm])
              
    for i in np.arange(0,Nx,1):
        for j in np.arange(i,Nx,1):
            F[i]=F[i]+simps(ge_intp[j*Nm:(j+1)*Nm]/(np.sqrt(1.0-xji[j,:]**2)*Eyxy[i,j,:]),dx=dx_f)
        F[i]=F[i]-1.0      
    pl.plot(F[0:Nx]+1.0,'r^')
    pl.show()
    # calculate the F2: equal length condition
    for i in np.arange(Nx): # this is y
        for j in np.arange(i+1): # this is x, x is faster
            F[i+Nx]=F[i+Nx]+simps(1.0/Exxy[i,j,:],dx=dx_f)
              

    pl.plot(F[0:100],'r^')
    pl.show()
    return F


def Exy_build(X):
    eps=0.0
    for i in np.arange(Nx): # this is y
        for j in np.arange(i+1): # this is x, x is faster
            Exy[i,j]=(4.0/3.0)*delta**(-2)*(X[j+Nx]+X[i+2*Nx]/X[i])
            Exy[i,j]=np.sqrt(Exy[i,j])  


    # the first four grid are replaced by anlytical solution
    for i in np.arange(0,5): # this is y
        for j in np.arange(i+1): # this is x, x is faster
            for k in np.arange(Nm):
                Exxy[i,j,k]=0.5*np.pi*np.sqrt(xi[i]**2-xij[j,k]**2+eps)
    for i in np.arange(5,Nx): # this is y
        intp=np.polyfit(xi[i-5:i+1], Exy[i,i-5:i+1],5)
        p10 = np.poly1d(intp)
        for j in np.arange(i+1): # this is x, x is faster
            Exxy[i,j,:]=p10(xij[j,:])
    #for i in np.arange(Nx): # this is x
    #    for j in np.arange(i,Nx): # this is y
    #        for k in np.arange(Nm):
    #            Eyxy[i,j,k]=0.5*np.pi*np.sqrt(xji[j,k]**2-xi[i]**2+eps)
    #



if __name__ == '__main__':
    res=[]
    #for size in [2.23]:
    for size in [6.122]:
        res.append((size,PolymerBrush_SSL(size)))
    print res

