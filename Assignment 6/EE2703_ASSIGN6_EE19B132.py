import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Default Values
n = 100
M = 5
nk = 500
u0 = 5
p = 0.25
Msig = 0.2

try:
    n = int(sys.argv[1])
    M = int(sys.argv[2])
    nk = int(sys.argv[3])
    u0 = float(sys.argv[4])
    p = float(sys.argv[5])
    Msig = float(sys.argv[6])
except:
    print('Simulating with Default Values.')

# Initializing Vectors
xx = np.zeros(n*M)
u = np.zeros(n*M)
dx = np.zeros(n*M)
I = []
X = []
V = []

for i in range(nk):
    # Updating the vectors
    ii = np.where(xx>0)[0]
    dx[ii] = u[ii] + 0.5
    xx[ii] = xx[ii] + dx[ii]
    u[ii] = u[ii] + 1
    # Checking if electrons reached anode and resetting vectors
    reached_anode = np.where(xx > n)[0]
    xx[reached_anode] = 0
    u[reached_anode] = 0
    dx[reached_anode] = 0
    # Checking if velocity more than theshold
    kk = np.where(u>u0)[0]
    ll = np.where(np.random.randn(len(kk))<=p)[0]
    kl = kk[ll]
    
    u[kl] = 0
    xx[kl] = xx[kl] - dx[kl]*np.random.rand()
    
    I.extend(xx[kl].tolist())
    
    m = int(np.random.randn()*Msig+M)
    # Injecting electrons in each turn
    free_slots = np.where(xx==0)[0]
    # Checking if number of free bins are greater than m
    if len(free_slots)>=m:
        xx[free_slots[np.random.randint(len(free_slots)):np.random.randint(len(free_slots))+m]] = 1
        u[free_slots[np.random.randint(len(free_slots)):np.random.randint(len(free_slots))+m]] = 0
    else:
        xx[free_slots] = 1
        u[free_slots] = 0
        
    remaining_electrons = np.where(xx>0)[0]
    X.extend(xx[remaining_electrons].tolist())
    V.extend(u[remaining_electrons].tolist())

# Plotting Electron Density and Intensity Histograms
fig, axes = plt.subplots(1,2, figsize = (15,5))
axes[0].hist(X,bins=np.arange(1,100), ec = 'black')
axes[0].title.set_text('Electron Density Histogram')
axes[0].set_xlabel('Bin Number')

axes[1].hist(I,bins=np.arange(1,100), ec = 'black')
axes[1].title.set_text('Light Intensity Histogram')
axes[1].set_xlabel('Bin Number')

plt.show()

# Plotting Electron Space Phase Plot
plt.plot(xx, u, '*')
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.title('Electron Phase Space Plot')
plt.show()

# Printing Intensity Table
a,bins,_=plt.hist(I,bins=np.arange(1,100),ec='black')
xpos=0.5*(bins[0:-1]+bins[1:])
d={'Position':xpos,'Count':a}
p=pd.DataFrame(d)
print(p)