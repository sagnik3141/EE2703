#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sympy
import pylab as p
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt


# In[2]:


def lowpass(R1,R2,C1,C2,G,Vi):
    s=sympy.symbols('s')
    A=sympy.Matrix([[0,0,1,-1/G],[-1/(1+s*R2*C2),1,0,0],     [0,-G,G,1],[-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
    b=sympy.Matrix([0,0,0,Vi/R1])
    
    V=A.inv()*b
    
    return A, b, V


# In[3]:


A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)
print('G=1000')
Vo=V[3]
print(Vo)
s=sympy.symbols('s')
w=p.logspace(0,8,801)
ss=1j*w
hf=sympy.lambdify(s,Vo,'numpy')
v=hf(ss)
plt.loglog(w,abs(v),lw=2)
plt.grid(True)
plt.xlabel(r'$\omega \rightarrow$')
plt.ylabel(r'$|H(\omega)| \rightarrow$')
plt.title('Magnitude Response of the Low Pass Filter')
plt.show()


# In[4]:


# Question 1
A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)

print ('G=1000')
Vo=V[3]
Vo=sympy.simplify(Vo)
display(Vo)

s,t=sympy.symbols("s t")
t=sympy.Symbol("t",positive=True)
n,d = sympy.fraction(Vo)
n_sp,d_sp=(np.array(sympy.Poly(j,s).all_coeffs(),dtype=float) for j in (n,d))

print(n_sp,d_sp)
ts=np.linspace(0,0.001,8001)
t,x,svec=sp.lsim(sp.lti(n_sp,d_sp),np.ones(len(ts)),ts)
# Plot the absolute step response
plt.plot(t,np.abs(x),lw=2)
plt.grid(True)
plt.xlabel('t')
plt.ylabel('Response')
plt.title('Unit Step Response')
plt.show()


# In[5]:


def highpass(R1,R2,C1,C2,G,Vi):
    s=sympy.symbols('s')
    A=sympy.Matrix([[s*(C1+C2)+1/R1,0,-s*C2,-1/R1],[0,G,0,-1],     [-s*C2,0,1/R2+s*C2,0],[0,0,-G,1]])
    b=sympy.Matrix([Vi*s*C1,0,0,0])
    V=A.inv()*b
    return (A,b,V)


# In[6]:


# Question 2
A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)

Vo=V[3]
Vo=sympy.simplify(Vo)
n,d = sympy.fraction(Vo)
n_sp,d_sp=(np.array(sympy.Poly(j,s).all_coeffs(),dtype=float) for j in (n,d))

# Function to simulate
ts=np.linspace(0,0.001,8001)
vi= np.sin(2000*np.pi*ts)+np.cos(2*10**6*np.pi*ts)
plt.plot(vi)
plt.xlabel('t')
plt.ylabel(r'$V_i$')
plt.title('Sum of Sinusoids')
plt.show()

t,x,svec=sp.lsim(sp.lti(n_sp,d_sp),vi,ts)
# Plot the lamdified values
plt.plot(t,x,lw=2)
plt.grid(True)
plt.xlabel('t')
plt.ylabel(r'$V_o$')
plt.title('Output for Low Pass Filter')
plt.show()

A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1)

Vo=V[3]
Vo=sympy.simplify(Vo)
n,d = sympy.fraction(Vo)
n_sp,d_sp=(np.array(sympy.Poly(j,s).all_coeffs(),dtype=float) for j in (n,d))

# Function to simulate
ts=np.linspace(0,0.001,8001)
vi= np.sin(2000*np.pi*ts)+np.cos(2*10**6*np.pi*ts)

t,x,svec=sp.lsim(sp.lti(n_sp,d_sp),vi,ts)
# Plot the lamdified values
plt.plot(t,x,lw=2)
plt.grid(True)
plt.xlabel('t')
plt.ylabel(r'$V_o$')
plt.title('Output for High Pass Filter')
plt.show()


# In[7]:


# Question 3
A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1)

print ('G=1000')
Vo=V[3]
print("Vo (transfer function)")
display(Vo)
Vo=sympy.simplify(Vo)
print("Vo (transfer function) Simplified")
display(Vo)
w=np.logspace(0,8,801)
ss=1j*w
hf=sympy.lambdify(s,Vo,"numpy")
v=hf(ss)

plt.loglog(w,abs(v),lw=2)
plt.grid(True)
plt.xlabel(r'$\omega$')
plt.ylabel(r'Magnitude')
plt.title('Magnitude Response of High Pass Filter')
plt.show()


# In[8]:


t = np.linspace(0,1e-3,100000)
decay=3e3
freq=1e7
vi = np.cos(freq*t)*np.exp(-decay*t) * (t>0)
plt.plot(t,vi)
plt.grid(True)
plt.xlabel('t')
plt.ylabel(r'$V_i$')
plt.title('High Frquency Damped Sinusoid')
plt.show()

A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1)

Vo=V[3]
Vo=sympy.simplify(Vo)
n,d = sympy.fraction(Vo)
n_sp,d_sp=(np.array(sympy.Poly(j,s).all_coeffs(),dtype=float) for j in (n,d))

t,x,svec=sp.lsim(sp.lti(n_sp,d_sp),vi,t)
plt.plot(t,x,lw=2)
plt.grid(True)
plt.xlabel('t')
plt.ylabel(r'$V_o$')
plt.title('Output for High Pass Filter')
plt.show()

A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)

Vo=V[3]
Vo=sympy.simplify(Vo)
n,d = sympy.fraction(Vo)
n_sp,d_sp=(np.array(sympy.Poly(j,s).all_coeffs(),dtype=float) for j in (n,d))

t,x,svec=sp.lsim(sp.lti(n_sp,d_sp),vi,t)
plt.plot(t,x,lw=2)
plt.grid(True)
plt.xlabel('t')
plt.ylabel(r'$V_o$')
plt.title('Output for Low Pass Filter')
plt.show()


# In[9]:


t = np.linspace(0,.5,100000)

decay=1e1
freq=1e3
vi = np.cos(freq*t)*np.exp(-decay*t) * (t>0)
plt.plot(t,vi)
plt.grid(True)
plt.xlabel('t')
plt.ylabel(r'$V_i$')
plt.title('Low Frquency Damped Sinusoid')
plt.show()

A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1)

Vo=V[3]
Vo=sympy.simplify(Vo)
n,d = sympy.fraction(Vo)
n_sp,d_sp=(np.array(sympy.Poly(j,s).all_coeffs(),dtype=float) for j in (n,d))

t,x,svec=sp.lsim(sp.lti(n_sp,d_sp),vi,t)
plt.plot(t,x,lw=2)
plt.grid(True)
plt.xlabel('t')
plt.ylabel(r'$V_o$')
plt.title('Output for High Pass Filter')
plt.show()

A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)

Vo=V[3]
Vo=sympy.simplify(Vo)
n,d = sympy.fraction(Vo)
n_sp,d_sp=(np.array(sympy.Poly(j,s).all_coeffs(),dtype=float) for j in (n,d))

t,x,svec=sp.lsim(sp.lti(n_sp,d_sp),vi,t)
plt.plot(t,x,lw=2)
plt.grid(True)
plt.xlabel('t')
plt.ylabel(r'$V_o$')
plt.title('Output for Low Pass Filter')
plt.show()


# In[12]:


# Question 5
A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1)
Vo=V[3]
Vo=sympy.simplify(Vo)
s,t=sympy.symbols("s t")
t=sympy.Symbol("t",positive=True)
n,d = sympy.fraction(Vo)
n_sp,d_sp=(np.array(sympy.Poly(j,s).all_coeffs(),dtype=float) for j in (n,d))

print(n_sp,d_sp)
ts=np.linspace(0,0.001,8001)
t,x,svec=sp.lsim(sp.lti(n_sp,d_sp),np.ones(len(ts)),ts)
# Plot the lamdified values
plt.plot(t,x,lw=2)
plt.grid(True)
plt.xlabel('t')
plt.ylabel(r'$V_o$')
plt.title('Unit Step Response')
plt.show()


# In[ ]:




