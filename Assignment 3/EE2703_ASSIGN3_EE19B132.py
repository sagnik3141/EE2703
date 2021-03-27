import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.special as sp

# Loading the Data
with open('fitting.dat') as f:
    lines = f.readlines()

# Extracting the Data as an array
for i in range(len(lines)):
    lines[i] = lines[i].strip().split()
    for l in range(len(lines[i])):
        lines[i][l] = float(lines[i][l])
    
sigma = np.logspace(-1,-3,9)

lines = np.array(lines)
data = lines.T
time = data[0]
data = data[1:] # 9 columns of data are the data to be fitted with varying degree of noise

# Original Function

def g(t, A, B):
    return A*sp.jn(2,t) + B*t

orig_func = [g(t, 1.05, -0.105) for t in time] # Original Funtion Values

# Plot of data to be fitted
legends = []
for i in range(len(data)):
    plt.plot(time, data[i])
    legends.append('Noise Std Dev = %1.3f'%(sigma[i]))
plt.plot(time, orig_func, color = 'black')
legends.append('Original Function')
plt.xlabel(r'$t$', size = 15)
plt.ylabel(r'$f(t)+n$', size = 15)
plt.title(r'Plot of Data to be Fitted')
plt.legend(legends)
plt.show()

# Plotting Error Bars
plt.errorbar(time[::5],data[0][::5],sigma[0],fmt='ro')
legends = []
legends.append('Original Function')
plt.plot(time, orig_func)
legends.append('Error Bars')
plt.legend(legends)
plt.xlabel(r'$t$')
plt.title(r'Error Bar and Original Function')
plt.show()

# Generating Matrix

def generate_matrix(time):
    x = sp.jn(2, time).T
    y = time.T
    M = np.c_[x,y]
    return M

A = np.arange(0, 2.1, 0.1)
B = np.arange(-0.2, 0.01, 0.01)

(x1, y1) = np.meshgrid(A, B) # Creating Meshgrid to plot contours

mse_error = np.zeros((len(A), len(B))) # Initialising the error matrix to be plotted

for i in range(len(A)):
    for k in range(len(B)):
        mse_error[i][k]+=(np.matmul((data[0].T - np.array([g(t, A[i], B[k]) for t in time]).T).T,
        data[0].T - np.array([g(t, A[i], B[k]) for t in time]).T))/101 # This line implements the error equation

# Plotting Contour Plot
fig, ax = plt.subplots()
CS = ax.contour(x1, y1, mse_error, levels =10)
ax.clabel(CS)
ax.scatter([1.05], [-0.105])
ax.set_xlabel(r'Parameter A')
ax.set_ylabel(r'Parameter B')
ax.set_title(r'Contour Plots of MSE Error')
plt.show()

true_parameters = [1.05, -0.105]
estimated_parameters=[]

for i in range(len(data)):
    estimated_parameters.append(scipy.linalg.lstsq(generate_matrix(time), data[i].T)[0]) # Using scipy to estimate A and B

estimated_parameters = np.array(estimated_parameters)

# Extracting estimated A and B
estimated_A = estimated_parameters.T[0].T
estimated_B = estimated_parameters.T[1].T
# Computing MSE
mse_A = (estimated_A - 1.05)**2
mse_B = (estimated_B + 0.105)**2

# Plotting MSE on a linear scale
plt.plot(sigma, mse_A, marker = 'o', markersize = 5, linestyle = 'dashed')
plt.plot(sigma, mse_B, marker = 'o', markersize = 5, linestyle = 'dashed')
plt.legend(['Error in A', 'Error in B'])
plt.title(r'Errors in A and B in linear scale')
plt.xlabel(r'Standard Deviation')
plt.ylabel(r'MS Error')
plt.show()

# Plotting MSE on a LogLog scale
plt.stem(sigma, mse_A, use_line_collection=True)
plt.stem(sigma, mse_B, 'r','ro', use_line_collection=True)
plt.legend(['Error in A', 'Error in B'])
plt.title(r'Errors in A and B in log scale')
plt.xlabel(r'Standard Deviation')
plt.ylabel(r'MS Error')
plt.xscale('log')
plt.yscale('log')
plt.show()





