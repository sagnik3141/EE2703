import numpy as np
import matplotlib.pyplot as plt
import sys
import mpl_toolkits.mplot3d.axes3d as p3
from pylab import *

nx = int(sys.argv[1])
ny = int(sys.argv[2])
radius = (float(sys.argv[3])/nx + float(sys.argv[3])/ny)/2
niter = int(sys.argv[4])

phi = np.zeros((ny, nx))

y = np.linspace(-0.5, 0.5, ny)
x = np.linspace(-0.5, 0.5, nx)
Y, X = np.meshgrid(y, x)

ii = np.where(X*X+Y*Y <= radius*radius)
phi[np.where(X*X+Y*Y <= radius*radius)]=1
errors = np.zeros(niter)

for i in range(niter):
    oldphi = phi.copy()
    phi[1:-1, 1:-1] = 0.25*(phi[1:-1,0:-2]+phi[1:-1,2:]+phi[0:-2,1:-1]+phi[2:,1:-1])
    phi[1:-1,0] = phi[1:-1,1]
    phi[1:-1,-1] = phi[1:-1,-2]
    phi[0,0:] = phi[1,0:]
    phi[-1,1:-1] = 0
    phi[np.where(X*X+Y*Y <= radius*radius)]=1
    errors[i] = np.max(np.abs(phi-oldphi))

# Error Analysis

plt.plot(errors)
plt.xlabel(r'Number of Iterations')
plt.ylabel(r'Error')
plt.title(r'Evolution of Errors vs Iterations')
plt.show()

plt.semilogy(np.arange(len(errors))[::50], errors[::50], 'ro')
plt.xlabel(r'Iterations')
plt.ylabel(r'Errors')
plt.title(r'Evolution of Errors vs Iterations in semilogy')
plt.show()

plt.loglog(np.arange(len(errors))[::50], errors[::50], 'ro')
plt.xlabel(r'Iterations')
plt.ylabel(r'Errors')
plt.title(r'Evolution of Errors vs Iterations in loglog')
plt.show()
    
def generate_mat(niter):
    a = np.ones((niter, 1))
    b = np.arange(0, niter)
    M = np.c_[a,b]

    return M

log_error = np.log(errors)
M1 = generate_mat(niter)

x1 = np.linalg.lstsq(M1, log_error)[0]

A1 = np.exp(x1[0])
B1 = x1[1]

M2 = generate_mat(niter-500)
M2[:, 1] += 500
x2 = np.linalg.lstsq(M2, log_error[500:])[0]

A2 = np.exp(x2[0])
B2 = x2[1]

errors_fit1 = A1*np.exp(B1*np.arange(niter))
errors_fit2 = A2*np.exp(B2*np.arange(niter))

plt.semilogy(np.arange(len(errors_fit1))[::50], errors_fit1[::50], 'ro')
plt.semilogy(np.arange(len(errors_fit2))[::50], errors_fit2[::50], 'bo')
plt.xlabel(r'Iterations')
plt.ylabel(r'Errors')
plt.title(r'Estimated Evolution of Errors vs Iterations in semilogy')
plt.legend(['Fit 1', 'Fit 2 (After 500 iters)'])
plt.show()

#3D Plot
fig1=plt.figure(4)
ax=p3.Axes3D(fig1)
plt.title('The 3-D surface plot of the potential')
surf = ax.plot_surface(Y, X, phi.T, rstride=1, cstride=1, cmap=cm.jet)
plt.xlabel(r'x')
plt.ylabel(r'y')
plt.show()

#Contour Plot
plt.contour(X,Y, phi.T)
plt.gca().invert_yaxis()
plt.title('Contour Plot of potential')
plt.xlabel(r'x')
plt.ylabel(r'y')
plt.plot(X[ii], Y[ii], 'ro')
plt.savefig('Figure_6.png')

Jx = np.zeros((ny,nx))
Jy = np.zeros((ny,nx))
Jx[:,1:-1] = 0.5*(phi[:,0:-2]-phi[:,2:])
Jy[1:-1,:] = 0.5*(phi[2:, :]-phi[0:-2,:])
fig,ax = plt.subplots()
fig = ax.quiver(Y[2:-1],X[::-1][2:-1],Jx[2:-1],Jy[2:-1],scale=5)
plt.plot(X[ii], Y[ii], 'ro')
plt.title("Current Density")
plt.xlabel(r'x')
plt.ylabel(r'y')
plt.show()