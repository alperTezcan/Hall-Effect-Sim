import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class StonerWohlfarthModel:
    def __init__(self, PhiH, K, Ms, PhiU, ThetaH, output_dir):
        self.PhiH = PhiH
        self.K = K
        self.Ms = Ms
        self.PhiU = PhiU
        self.ThetaH = ThetaH
        self.output_dir = output_dir
        self.Mparam0 = [np.pi / 2, 0.0]  # Initial guess for theta, phi (spherical angles)
        self.M = []  # To store magnetization results
        self.setup_output_directory()

    def setup_output_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def magnetization_vector(self, theta, phi):
        """
        Convert spherical coordinates (theta, phi) to a Cartesian vector.
        """
        return np.array([
            np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(theta)
        ])

    def applied_field_vector(self, H):
        """
        Compute the applied magnetic field vector in Cartesian coordinates.
        """
        return H * np.array([
            np.cos(self.PhiH) * np.sin(self.ThetaH),
            np.sin(self.PhiH) * np.sin(self.ThetaH),
            np.cos(self.ThetaH)
        ])

    def energie(self, Mparam, H):
        """
        Total energy function combining Zeeman, demagnetizing, and anisotropy energies.
        """
        theta, phi = Mparam
        M = self.magnetization_vector(theta, phi)
        H_vec = self.applied_field_vector(H)

        # Energy terms
        Ez = -np.dot(M, H_vec)  # Zeeman energy
        Ed = 0.5 * self.Ms * M[2]**2  # Demagnetizing energy (z-axis anisotropy)
        Ean = -0.5 * self.K * np.cos(phi - self.PhiU)**2  # Anisotropy energy
        
        return Ez + Ed + Ean

    def minimize_energy(self, H):
        """
        Minimize the total energy for a given field strength H.
        """
        result = minimize(
            self.energie,
            self.Mparam0,  # Initial guess for theta, phi
            args=(H,),
            method='Nelder-Mead'
        )
        self.Mparam0 = result.x  # Update the initial guess for the next iteration
        return self.Mparam0

    def calculate_magnetization(self):
        """
        Sweep over the magnetic field and minimize the energy at each step.
        Store the magnetization components (Mx, My, Mz).
        """
        field_sweep = np.concatenate((np.arange(-5, 5.1, 0.1), np.arange(5, -5.1, -0.1)))

        for H in field_sweep:
            theta, phi = self.minimize_energy(H)
            M_vec = self.magnetization_vector(theta, phi)

            # Store the field and magnetization components
            self.M.append([H, M_vec[0], M_vec[1], M_vec[2]])

        self.M = np.array(self.M)

    def save_results(self):
        """
        Save the magnetization results to a text file.
        """
        filename = f"PhiH={self.PhiH},K={self.K},Ms={self.Ms},PhiU={self.PhiU},ThetaH={self.ThetaH}.txt"
        filepath = os.path.join(self.output_dir, filename)
        np.savetxt(filepath, self.M, header="H Mx My Mz")
        print(f"Results saved to {filepath}")

    def plot_magnetization(self):
        """
        Plot Mx, My, Mz as a function of the magnetic field H.
        """
        plt.figure(figsize=(12, 8))

        plt.subplot(1, 3, 1)
        plt.plot(self.M[:, 0], self.M[:, 1], 'b-', label='Mx')
        plt.xlabel('H')
        plt.ylabel('Mx')
        plt.title('Mx vs H')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(self.M[:, 0], self.M[:, 2], 'r-', label='My')
        plt.xlabel('H')
        plt.ylabel('My')
        plt.title('My vs H')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(self.M[:, 0], self.M[:, 3], 'g-', label='Mz')
        plt.xlabel('H')
        plt.ylabel('Mz')
        plt.title('Mz vs H')
        plt.grid(True)
        plt.legend()

        # Save the plot
        plot_filepath = os.path.join(self.output_dir, "Magnetization_vs_H.png")
        plt.tight_layout()
        plt.savefig(plot_filepath)
        plt.show()
        print(f"Plot saved to {plot_filepath}")

if __name__ == "__main__":
    # Parameters initialization
    PhiH = 0.0
    K = 0.0005
    Ms = 1.0
    PhiU = 140 * np.pi / 180  # Anisotropy axis angle in radians
    ThetaH = np.pi / 2  # External field angle in radians

    # User input for output directory
    output_dir = input("Enter the directory to save the results: ")

    # Create the Stoner-Wohlfarth model instance
    model = StonerWohlfarthModel(PhiH, K, Ms, PhiU, ThetaH, output_dir)

    # Calculate magnetization and save results
    model.calculate_magnetization()
    model.save_results()

    # Plot results
    model.plot_magnetization()
