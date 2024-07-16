import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def magnetic_field(r, m , r0):
    """
    Calculate the magnetic field vector at position r due to a dipole with moment m.
    
    Parameters:
    r (numpy.ndarray): Position vector (shape: 3).
    m (float): Magnitude of the magnetic moment.
    
    Returns:
    numpy.ndarray: Magnetic field vector at r (shape: 3).
    """
    x, y, z = r-r0 
    norm_r = np.linalg.norm(r)
    h = (m / (4 * np.pi * norm_r**5)) * np.array([2 * x**2 - y**2 - z**2, 3 * x * y, 3 * x * z])
    return h

# Define the magnetic moment magnitude
m_magnitude = 1.0

# Define the grid for visualization
x = np.linspace(-10, 10, 20)
y = np.linspace(-10, 10, 20)
z = np.linspace(-10,10,20)
X, Y, Z = np.meshgrid(x, y, z) 

r0 = np.array([5,6,0])

# Calculate the magnetic field vectors
U, V, W = np.zeros(X.shape), np.zeros(Y.shape), np.zeros(Z.shape)
for i in range(X.shape[0]):
    for j in range(Y.shape[1]):
        for k in range(Z.shape[2]):
            r = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
            h = magnetic_field(r, m_magnitude , r0)
            U[i, j, k], V[i, j, k], W[i, j, k] = h

# Plot the magnetic field vectors
fig = plt.figure()
ax = fig.add_subplot(111) #, projection='2d'
ax.quiver(X[:,:,0], Y[:,:,0], U[:,:,0], V[:,:,0])
ax.scatter([r0[0]] , [r0[1]] , s = 10 , c='r')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Magnetic Field Vectors')

plt.show()

