import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# Function to calculate the total energy
def Energie(Mparam, H, ThetaH, PhiH, K, PhiU, Ms):
    theta, phi = Mparam
    
    # Magnetization vector M
    M = np.array([
        np.cos(phi) * np.sin(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(theta)
    ])
    
    # Applied magnetic field vector H
    H_vec = H * np.array([
        np.cos(PhiH) * np.sin(ThetaH),
        np.sin(PhiH) * np.sin(ThetaH),
        np.cos(ThetaH)
    ])
    
    # Energy components
    Ez = -np.dot(M, H_vec)  # Zeeman energy
    Ed = 0.5 * Ms * M[2]**2  # Demagnetizing energy (z-axis anisotropy)
    Ean = -0.5 * K * np.cos(phi - PhiU)**2  # Anisotropy energy
    
    # Total energy
    E = Ez + Ed + Ean
    return E

# Parameters initialization
PhiH = 0.0
K = 0.0005
Ms = 1.0
PhiU = 140 * np.pi / 180  # Anisotropy axis angle in radians
Mparam0 = [np.pi / 2, 0.0]  # Initial guess for [theta, phi]
ThetaH = np.pi / 2  # External field angle in radians

# Result storage
M = []

# Sweep in positive and negative fields
for H in np.concatenate((np.arange(-5, 5.1, 0.1), np.arange(5, -5.1, -0.1))):
    # Minimization using scipy.optimize.minimize
    result = minimize(Energie, Mparam0, args=(H, ThetaH, PhiH, K, PhiU, Ms), method='Nelder-Mead')
    
    # Update the magnetization angles for the next iteration
    Mparam0 = result.x
    
    # Store the field and magnetization components
    Mx = np.cos(Mparam0[1]) * np.sin(Mparam0[0])
    My = np.sin(Mparam0[1]) * np.sin(Mparam0[0])
    Mz = np.cos(Mparam0[0])
    
    M.append([H, Mx, My, Mz])

# Convert results to numpy array
M = np.array(M)

# Prompt the user to select a directory to save the results
output_dir = input("Enter the directory to save the results: ")
os.makedirs(output_dir, exist_ok=True)

# Create a filename using the simulation parameters
filename = f"PhiH={PhiH},K={K},Ms={Ms},PhiU={PhiU},ThetaH={ThetaH}.txt"
filepath = os.path.join(output_dir, filename)

# Save the results to a text file
np.savetxt(filepath, M, header="H Mx My Mz")

print(f"Results saved to {filepath}")

# Plot Mx, My, Mz with respect to H
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.plot(M[:, 0], M[:, 1], 'b-', label='Mx')
plt.xlabel('H')
plt.ylabel('Mx')
plt.title('Mx vs H')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(M[:, 0], M[:, 2], 'r-', label='My')
plt.xlabel('H')
plt.ylabel('My')
plt.title('My vs H')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(M[:, 0], M[:, 3], 'g-', label='Mz')
plt.xlabel('H')
plt.ylabel('Mz')
plt.title('Mz vs H')
plt.grid(True)
plt.legend()

# Save the plot
plot_filepath = os.path.join(output_dir, "Magnetization_vs_H.png")
plt.tight_layout()
plt.savefig(plot_filepath)
plt.show()

print(f"Plot saved to {plot_filepath}")