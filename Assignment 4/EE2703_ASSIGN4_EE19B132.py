# Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import lstsq

# Definind the functions to be approximated
def exp_func(x):
    return np.exp(x)

def coscos_func(x):
    return np.cos(np.cos(x))

# Evaluating the functions in the range (-2pi, 4pi)
y = np.linspace(-2*np.pi, 4*np.pi, 100)
exp_values = exp_func(y)
coscos_values = coscos_func(y)

# Plotting the original functions
plt.plot(y, exp_values)
plt.ylabel(r'$exp(x)$', size = 12)
plt.xlabel(r'$x$', size = 12)
plt.title(r'Plot of $exp(x)$')
plt.yscale('log')
plt.grid(True)
plt.show()

plt.plot(y, coscos_values)
plt.ylabel(r'$cos(cos(x))$', size = 12)
plt.xlabel(r'$x$', size = 12)
plt.title(r'Plot of $cos(cos(x))$')
plt.grid(True)
plt.show()

# Evaluating the fourier coefficients of exp(x) using direct integration
exp_coeff = []
exp_coeff.append((1/(2*np.pi))*quad(lambda x: np.exp(x), 0, 2*np.pi)[0])
for i in range(1,26):
    exp_coeff.append((1/np.pi)*quad(lambda x: np.exp(x)*np.cos(i*x), 0, 2*np.pi)[0])
    exp_coeff.append((1/np.pi)*quad(lambda x: np.exp(x)*np.sin(i*x), 0, 2*np.pi)[0])

# Plotting the coefficients of exp on semilog
plt.scatter(np.arange(0, len(exp_coeff)), np.abs(exp_coeff), color = 'red')
plt.yscale('log')
plt.xlabel(r'$n$', size = 12)
plt.ylabel(r'Coefficients', size = 12)
plt.title(r'Fourier Coefficients of $exp(x)$ on a semilog scale')
plt.grid(True)
plt.show()

# Plotting the coefficients of exp on loglog
plt.loglog(np.arange(0, len(exp_coeff)), np.abs(exp_coeff), 'ro')
plt.xlabel(r'$n$', size = 12)
plt.ylabel(r'Coefficients', size = 12)
plt.title(r'Fourier Coefficients of $exp(x)$ on a loglog scale')
plt.grid(True)
plt.show()

# Evaluating the fourier coefficients of cos(cos(x)) using direct integration
coscos_coeff = []
coscos_coeff.append((1/(2*np.pi))*quad(lambda x: np.cos(np.cos(x)), 0, 2*np.pi)[0])
for i in range(1,26):
    coscos_coeff.append((1/np.pi)*quad(lambda x: np.cos(np.cos(x))*np.cos(i*x), 0, 2*np.pi)[0])
    coscos_coeff.append((1/np.pi)*quad(lambda x: np.cos(np.cos(x))*np.sin(i*x), 0, 2*np.pi)[0])

# Plotting the coefficients of cos(cos(x)) on semilog
plt.scatter(np.arange(0, len(coscos_coeff)), np.abs(coscos_coeff), color = 'red')
plt.yscale('log')
plt.xlabel(r'$n$', size = 12)
plt.ylabel(r'Coefficients', size = 12)
plt.title(r'Fourier Coefficients of $cos(cos(x))$ on a semilog scale')
plt.grid(True)
plt.show()

# Plotting the coefficients of cos(cos(x)) on loglog
plt.loglog(np.arange(0, len(coscos_coeff)), np.abs(coscos_coeff), 'ro')
plt.xlabel(r'$n$', size = 12)
plt.ylabel(r'Coefficients', size = 12)
plt.title(r'Fourier Coefficients of $cos(cos(x))$ on a loglog scale')
plt.grid(True)
plt.show()

# Least Squares

x = np.linspace(0, 2*np.pi, 401) # Defining the range
x = x[:-1]

def generate_matrix():  # Generating the matrix
    A = np.zeros((400, 51))
    A[:,0] = 1

    for i in range(1, 51):
        if i%2 == 0:
            A[:,i] = np.sin(x*((i+1)//2))
        else:
            A[:,i] = np.cos(x*((i+1)//2))

    return A

A = generate_matrix()

# Least Squares for exp()

b_exp = exp_func(x) # b_exp holds the function values in the interval

c_exp = lstsq(A, b_exp)[0] # Least Sq estimate for exp()

# Plotting Least Sq Coefficients of exp() on semilog scale
plt.scatter(np.arange(0, len(exp_coeff)), np.abs(exp_coeff), color = 'red')
plt.scatter(np.arange(0, len(c_exp)), np.abs(c_exp), color = 'green')
plt.legend(['True Coefficients', 'Predicted Coefficients'])
plt.yscale('log')
plt.xlabel(r'$n$', size = 12)
plt.ylabel(r'Coefficients', size = 12)
plt.title(r'Fourier Coefficients of $exp(x)$ on a semilog scale using Least Squares')
plt.grid(True)
plt.show()

# Plotting Least Sq Coefficients of exp() on loglog scale
plt.loglog(np.arange(0, len(exp_coeff)), np.abs(exp_coeff), 'ro')
plt.loglog(np.arange(0, len(c_exp)), np.abs(c_exp), 'go')
plt.legend(['True Coefficients', 'Predicted Coefficients'])
plt.xlabel(r'$n$', size = 12)
plt.ylabel(r'Coefficients', size = 12)
plt.title(r'Fourier Coefficients of $exp(x)$ on a loglog scale using Least Squares')
plt.grid(True)
plt.show()

# Least Squares for coscos

b_coscos = coscos_func(x) # Function values

c_coscos = lstsq(A, b_coscos)[0] # Estimated Coeff

# Plotting Least Sq Coefficients of cos(cos()) on semilog scale
plt.scatter(np.arange(0, len(coscos_coeff)), np.abs(coscos_coeff), color = 'red')
plt.scatter(np.arange(0, len(c_coscos)), np.abs(c_coscos), color = 'green')
plt.legend(['True Coefficients', 'Predicted Coefficients'])
plt.yscale('log')
plt.xlabel(r'$n$', size = 12)
plt.ylabel(r'Coefficients', size = 12)
plt.title(r'Fourier Coefficients of $cos(cos(x))$ on a semilog scale using Least Squares')
plt.grid(True)
plt.show()

# Plotting Least Sq Coefficients of cos(cos()) on loglog scale
plt.loglog(np.arange(0, len(coscos_coeff)), np.abs(coscos_coeff), 'ro')
plt.loglog(np.arange(0, len(c_coscos)), np.abs(c_coscos), 'go')
plt.legend(['True Coefficients', 'Predicted Coefficients'])
plt.xlabel(r'$n$', size = 12)
plt.ylabel(r'Coefficients', size = 12)
plt.title(r'Fourier Coefficients of $cos(cos(x))$ on a loglog scale using Least Squares')
plt.grid(True)
plt.show()


# Finding Deviation for exp(x)

exp_dev = np.abs(exp_coeff - c_exp)
coscos_dev = np.abs(coscos_coeff - c_coscos)
max_exp_dev = np.max(exp_dev)
max_coscos_dev = np.max(coscos_dev)

print(f'Max Deviation for exp() is {max_exp_dev}')
print(f'Max Deviation for cos(cos()) is {max_coscos_dev}')

# Computing Ac for exp()

ac = np.matmul(A, c_exp)
exp_values = exp_func(x)

# Plotting the reconstructed exp() function
plt.scatter(x, exp_values, color = 'red')
plt.scatter(x, ac, color = 'green')
plt.legend(['True Values', 'Predicted Values'])
plt.yscale('log')
plt.xlabel(r'$x$', size = 12)
plt.ylabel(r'$exp(x)$', size = 12)
plt.title(r'Plot of true value of $exp(x)$ and Predicted Value')
plt.grid(True)
plt.show()

# Computing Ac for cos(cos(x))

ac_coscos = np.matmul(A, c_coscos)
coscos_values = coscos_func(x)

# Plotting the reconstructed cos(cos()) funtion
plt.scatter(x, coscos_values, color = 'red')
plt.scatter(x, ac_coscos, color = 'green')
plt.legend(['True Values', 'Predicted Values'])
plt.yscale('log')
plt.xlabel(r'$x$', size = 12)
plt.ylabel(r'$cos(cos(x))$', size = 12)
plt.title(r'Plot of true value of $cos(cos(x))$ and Predicted Value')
plt.grid(True)
plt.show()