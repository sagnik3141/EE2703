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

def generate_mat(niter):
    a = np.ones((niter, 1))
    b = np.arange(0, niter)
    M = np.c_[a,b]

    return M

log_error = np.log(errors)
M = generate_mat(niter)

x = np.linalg.lstsq(M, log_error)[0]

A = np.exp(x[0])
B = x[1]

print(A)
print(B)

plt.loglog(np.arange(niter), errors)
plt.show()

fig1=plt.figure(4)
ax=p3.Axes3D(fig1)
plt.title('The 3-D surface plot of the potential')
surf = ax.plot_surface(Y, X, phi.T, rstride=1, cstride=1, cmap=cm.jet)
plt.show()