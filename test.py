import numpy as np
import matplotlib.pyplot as plt

def dipole(m, r, r0):
    """
    Calculation of field B in point r. B is created by a dipole moment m located in r0.
    """
    # Calculate R and norm_R
    R = np.subtract(np.transpose(r), r0).T
    norm_R = np.sqrt(np.einsum("i...,i...", R, R)) 

    # Calculate A (replace x, y, z with appropriate values)
    x, y, z = r
    A = np.array([2*x**2 - y**2 - z**2, 3*x*y, 3*x*z])

    # Compute magnetic field B
    norm_m = np.sqrt(np.einsum("i...,i...", m, m))
    B = np.array((norm_m / ((4 * np.pi) * norm_R**5)) * A)
    B *= 1e-7   # Permeability of vacuum: 4*pi*10^(-7)
    return B

X = np.linspace(-10, 10, 100)
Y = np.linspace(-10, 10, 100)
Z = np.zeros(100)
X, Y, Z = np.meshgrid(X, Y, Z) 

Bx , By, _ = dipole([1,0,0] , r=[X,Y,Z],r0=[25,30,0])

# Plot magnetic field using streamplot
plt.figure(figsize=(8, 8))
plt.streamplot(X[:,:,0], Y[:,:,0], Bx[:,:,0], By[:,:,0]) 

plt.scatter([25] , [30] , c='r' ,  s=10)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Magnetic Field Streamlines')
plt.show()  





