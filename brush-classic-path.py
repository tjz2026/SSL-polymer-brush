#-*- coding: utf-8 -*-
"""
Brush Stong Stretching Theory
Ref : The Journal of Chemical Physics 117, 2351 (2002); doi: 10.1063/1.1487819
=========
:copyright: (c) 2016 by Jiuzhou Tang
:license: BSD, see LICENSE for more details.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as pl
from numpy.linalg import inv
import scipy as sc
from scipy.integrate import simps
import time
import scipy.optimize # nonlinear solver
from sys import exit

def PolymerBrush_SST():
    global Nx # spatial grid number
    global Lx # half of the box size [0~2Lx], Lx is scaled by aN^1/2, i.e. Re
    global dx # grid spacing
    global z_i # the spatial mesh array 
    global theta_i # the spatial angle mesh array 
    global zm # the maxium extend of brush
    global zm_indx # the maxium extend of brush
    global V0_ini # the initial-initial verlocity 
    global V0 # the initial verlocity
    global Vz # verlocity at position z
    global Wz # external potential W(z)
    global gz # end point distribution function g(z)
    global Sz # the dimensionless speed at z with intial position at z0
    global Phi # the segment distribution for a chain starting from z0,Phi(z,z0)
    global phiz # local density \phi(z)
    global fo  # free energy f0(z0)
    global feo  # free energy fe,0(z0)
    global fwo  # free energy fw,0(z0)

    Nx=100
    Lx=4.0 # Lx/Re
    dx=2*Lx/(Nx+1) # the starting and ending point is 0 and 2L
    z_i=np.zeros(Nx)
    sqr=np.zeros((Nx,Nx))
    theta_i=np.zeros(Nx)
    for i in np.arange(Nx):
        z_i[i]=(i+1)*dx
    theta_i[:]=(np.arange(Nx)+1)*(0.5*np.pi)/(Nx+1)
    V0=np.zeros(Nx)
    V0_ini=np.zeros(Nx)
    Vz=np.zeros(Nx)
    Wz=np.zeros(Nx)
    zm=np.zeros(Nx)
    zm_indx=np.arange(Nx)
    feo=np.zeros(Nx)
    fwo=np.zeros(Nx)
    Sz=np.zeros((Nx,Nx))
    Phi=np.zeros((Nx,Nx))
    init_w() # intialize w
    iter_max=500
    # the outter self-consistent iteration for W(z)
    for i in np.arange(iter_max):
        W_to_phiz() # calculate the local density phi(z), which needs g(z)
        update_w() # update the external poential W(z)
        convg=converge() # check if the imcompressibility is satisfied, i.e. converged
        if convg: 
            exit(0)

    print "converge?",convg
    pl.plot(gz[:],'r')
    pl.show()
    return

def init_w(): # intialize the extenal potential w(z)
    global Wz
    for i in np.arange(Nx):
        if (i<=Nx/2-1) :
            #z2=(i*dx)**2
            z2=z_i[i]**2
        else :
            #z2=((Nx-i)*dx)**2
            z2=(2*Lx-z_i[i])**2 
        Wz[i]=(-3*np.pi**2/8.0)*z2
    #pl.plot(Wz[:]/(Lx**2),'r')
    #pl.show()
    intp_w=np.polyfit(z_i[0:Nx/2],Wz[0:Nx/2],4)
    pw = np.poly1d(intp_w)
    print "W init,pw=",pw
    return

def update_w(): # update the external potential 
    global Wz
    lambda_t=0.1
    for i in np.arange(Nx):
        Wz[i]=(1.0-lambda_t)*Wz[i]+lambda_t*(phiz[i]+phiz[Nx-i-1]-1.0-Wz[i])
    return

def converge():
    error=0.0
    tol=1.0e-3
    for i in np.arange(Nx):
        error=error+np.abs(phiz[i]+phiz[Nx-i-1]-1.0)
    error=error/Nx
    if error<tol:
        convg=True
    else :
        convg=False
    return convg

def cal_zm(): # we must realize that in the physics sense, Sz can not be less than zero,
# and it can not overcome the poential barrier. must treat Sz carefully.
    global Sz
    global zm
    global zm_indx
    Sz[:,:]=0.0 
    eps=1.0e-3
    zm_indx[:]=-1
    for i in np.arange(Nx): # Z0
        if V0[i]<-eps: 
            for j in np.arange(i,-1,-1): # Z<=Z0
                Sz[i,j]=V0[i]**2-(2.0/3.0)*(Wz[i]-Wz[j])
                if Sz[i,j]<-eps: raise ValueError('Z<=Z0, V0<0,should accerate,Wz is not propriate')
                Sz[i,j]=np.sqrt(Sz[i,j])
            for j in np.arange(i+1,Nx,1): # Z>Z0
                Sz[i,j]=V0[i]**2-(2.0/3.0)*(Wz[i]-Wz[j])
                #print "Sz[i,j]",Sz[i,j],i,j
                if Sz[i,j]<eps:
                    zm_indx[i]=j
                    zm[i]=z_i[j]
                    break
                Sz[i,j]=np.sqrt(Sz[i,j])
        else : # if V0[i]>=0, then zm[i]=z_i[i] 
            for j in np.arange(i+1): # Z<=Z0
                Sz[i,j]=V0[i]**2-(2.0/3.0)*(Wz[i]-Wz[j])
                if Sz[i,j]<-eps:
                    print "Sz^2<0,V0,Wz[i],Wz[j]",V0[i]**2,(2/3.0)*(Wz[i]-Wz[j]),Sz[i,j],i,j,Wz[i],Wz[j] 
                    raise ValueError('Z<=Z0, V0>0, Wz is not propriate')
                Sz[i,j]=np.sqrt(Sz[i,j])
            
            zm[i]=z_i[i]
            zm_indx[i]=i
        if zm[i] <z_i[0] or zm[i]>z_i[Nx-1] : raise ValueError('Unrealistic value of zm',zm_indx[i],zm[i],i,z_i[i],V0[i])
        if zm_indx[i]<0 or zm_indx[i]>Nx-1  : raise ValueError('Unrealistic value of zm_indx',zm_indx[i],zm[i],i,V0[i])
        if V0[i]<0.0 and zm[i]>z_i[Nx/2] :
            print "V0[i],zm[i]",V0[i],zm[i],i
            raise ValueError('negative V0 too large!')
        if V0[i]>0.0 and zm[i]>z_i[i] :
            print "V0[i],zm[i]",V0[i],zm[i],i
            raise ValueError('zm is wrong')

    #for i in np.arange(Nx):
    #    print "i,V0,z_i[i],zm[i],zm_indx[i]",i,V0[i],z_i[i],zm[i],zm_indx[i]
    return zm,zm_indx





def cal_v0():
    global V0,zm,zm_indx
    iter_max=300
    tol_v0=1.0e-3
    zm,zm_indx=cal_zm()
    for i in np.arange(iter_max):
        zm,zm_indx=cal_zm()
        err=update_v0()
        print "iter for v0",i,err 
        if err<tol_v0 : 
            pl.plot(V0[:]/Lx,'r^')
            pl.show()  
            break
    pl.plot(V0[:]/Lx,'r^')
    pl.plot(V0_ini[:]/Lx,'b^')
    pl.show()
    np.savetxt('v0.dat',V0[:].T)  
    return 

def cal_phi():
    global Phi
    for i in np.arange(Nx): # Z0
        for j in np.arange(Nx): # Z
            if j<i: 
                Phi[i,j]=1.0/Sz[i,j] 
            elif  j>=i and j<zm_indx[i]: 
                Phi[i,j]=2.0/Sz[i,j] 
            else :
                Phi[i,j]=0.0
    
    return 
   


def update_v0():
    global V0
    lambda_v0=0.05
    cal_phi()
    f=np.zeros(Nx)
    g=np.zeros(Nx)
    sqr=np.zeros((Nx,Nx))
    for i in np.arange(Nx):
        for j in np.arange(zm_indx[i]+1):
            sqr[i,j]=np.sqrt(zm[i]**2-z_i[j]**2)
    F=np.zeros(Nx)
    #pl.plot(V0[:],'r^')
    #pl.show()
    #pl.plot(zm[:],'b^')
    #pl.show()
    #for i in np.arange(2,Nx,1):
    Nx_s=100
    Phi_s=np.zeros(Nx_s)
    Sz_s=np.zeros(Nx_s)
    x_s=np.zeros(Nx_s)
    V0_s=np.zeros(Nx_s)
    Wz_s=np.zeros(Nx_s)
    intp_v0=np.polyfit(z_i[0:15],V0[0:15],4)
    pv0 = np.poly1d(intp_v0)
    intp_W=np.polyfit(z_i[0:15],Wz[0:15],4)
    pw = np.poly1d(intp_W)
    #print "pw",pw
    #print "pv0",pv0
    for i in np.arange(2,15,1): # doing interpolation for the first few grids
        intp_v0=np.polyfit(z_i[0:7],V0[0:7],4)
        pv0 = np.poly1d(intp_v0)
        intp_W=np.polyfit(z_i[0:7],Wz[0:7],4)
        pw = np.poly1d(intp_W)
        for j in np.arange(Nx_s):
            x_s[j]=(zm[i]/(Nx_s+1.0))*(j+1)
            V0_s[j]=pv0(x_s[j])
            Wz_s[j]=pw(x_s[j])
            if V0_s[j]**2-(2.0/3.0)*(Wz[i]-Wz_s[j])<0 :
               print "V0_s[:],",V0_s[j]**2,-(2.0/3.0)*(Wz[i]-Wz_s[j]),i,j
               print "Wz_s[:] should be,",-(3.0/8.0)*np.pi**2*x_s[j]**2
               print "Wz_s[:] is,",Wz_s[j]
               print "Wz[i] is,",Wz[i]
               print "x_s[j],z_i[i]",x_s[j],z_i[i]
               print "zm,zm_indx",zm[i],zm_indx[i]
            Sz_s[j]=np.sqrt(V0_s[j]**2-(2.0/3.0)*(Wz[i]-Wz_s[j]))
            if x_s[j]<z_i[i]:
                Phi_s[j]=1.0/Sz_s[j]
            else :
                Phi_s[j]=2.0/Sz_s[j]
        F[i]=simps(Phi_s[:],dx=x_s[0])-1.0+0.09528
        V0[i]=(1.0-lambda_v0)*V0[i]+lambda_v0*(F[i]+V0[i])
        #print "F[i],i<8",F[i],i 
    for i in np.arange(15,Nx,1):
        if i<Nx/2: # Z0<Lx
            #intp=np.polyfit(z_i[0:zm_indx[i]], Phi[i,0:zm_indx[i]]*sqr[i,0:zm_indx[i]],3)
            intp=np.polyfit(z_i[0:i-5], Phi[i,0:i-5]*sqr[i,0:i-5],3)
            #intp=np.polyfit(z_i[0:i], Phi[i,0:i]*sqr[i,0:i],3)
            #intp=np.polyfit(z_i[0:zm_indx[i]], Phi[i,0:zm_indx[i]]*sqr[i,0:zm_indx[i]],3)
            p10 = np.poly1d(intp)
            g=p10(zm[i]*theta_i[:])
            F[i]=simps(g[:],dx=theta_i[0])-1.0
            if np.abs(F[i])>0.3 :
                print "p10",p10   
                print "Wz",Wz[i]-Wz[0:zm_indx[i]]
                print "Phi[i,:]",Phi[i,0:zm_indx[i]]
                print "sqr[i,:]",sqr[i,0:zm_indx[i]]
                print "Phi[i,:]*sqr[i,:]",Phi[i,0:zm_indx[i]]*sqr[i,0:zm_indx[i]]
                print "Sz[i,:]/sqr[i,:]",Sz[i,0:zm_indx[i]]/sqr[i,0:zm_indx[i]]
                print "F[i],i,V0,zm[i],z_i[i]",F[i],i,V0[i],zm[i],z_i[i],zm_indx[i]
                raise ValueError('F[i] not accurate for z<Lx')
            V0[i]=(1.0-lambda_v0)*V0[i]+lambda_v0*(F[i]+V0[i])
        else : # Z0>=Lx
            F[i]=simps(Phi[i,:],dx=z_i[0])-1.0
            V0[i]=(1.0-lambda_v0)*V0[i]+lambda_v0*(F[i]+V0[i])
    err=np.sqrt(np.sum(F[50:100]**2)/Nx)
    #print "F[:]",F[:]
    print "F[57],V[57]",F[57],V0[57]
    print "F[87],V[87]",F[87],V0[87]
    print "F[17],V[17]",F[17],V0[17]
    return err    
            


def init_V0():
    global V0
    eps=0.30
    p=0.0
    for i in np.arange(Nx/2):
        #V0[i]=-0.2*np.sin(np.pi*(i+1)/(Nx/2-1))*Lx
        V0[i]=0.0
    for i in np.arange(Nx/2,Nx,1): 
        #V0[i]=V0[Nx/2-1]+((i-Nx/2)/(Nx/2))*2.0*Lx+eps+5.0*np.sin(0.5*np.pi*(i-Nx/2)/(Nx/2))
        p=Lx+0.3333*(z_i[i]-Lx)+(2.0/(27*Lx))*(z_i[i]-Lx)**2
        V0[i]=np.pi*0.5*np.sqrt(z_i[i]**2-p**2)+eps
        #print "-p**2+z_i[i]**2",-p**2+z_i[i]**2,i
    #pl.plot(V0[:]/Lx,'r^')
    #pl.show()
    #V0[:]=np.loadtxt('v0.dat').T
    V0_ini[:]=V0[:]  
    return


def W_to_phiz(): # calculate the density from given W(z), also a self-consistent iteration,needs
# to determine the value of V0(z0)
    global Phi,V0
    global feo,fwo
    init_V0() # initialize V0(z0)
    V0=cal_v0() # determine the intial verlocity z0 according to equal time constrain
    cal_phi()
    # calculate g(y)
    for i in np.arange(4,Nx,1):
        feo[i]=1.5*simps(1.0/Phi[i,0:zm_indx[i]],dx=z_i[1]-z_i[0])
        fwo[i]=1.5*simps(Phi[i,0:zm_indx[i]]*Wz[0:zm_indx[i]],dx=z_i[1]-z_i[0])
    return          

    

if __name__ == '__main__':
    PolymerBrush_SST()






