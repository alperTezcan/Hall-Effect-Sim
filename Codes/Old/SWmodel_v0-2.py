import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class StonerWohlfarthModel:
    def __init__(self, theta, phi, ani_coef, thetaAni, phiAni):
        # Convert degrees to radians
        self.theta = np.radians(theta)
        self.phi = np.radians(phi)
        self.ani_coef = ani_coef
        self.thetaAni = np.radians(thetaAni)
        self.phiAni = np.radians(phiAni)
        
        self.setup_constants()
        self.setup_output_directory()

    def setup_constants(self):
        # Constants and applied fields
        self.R = 1
        self.DeltaR = 0.1
        self.A = 19.36
        self.Rahe = 0.0108

        self.Hstart = 6
        self.Hstep = 5 / 1000
        self.H = np.concatenate((np.arange(self.Hstart, -self.Hstart - self.Hstep, -self.Hstep), 
                                 np.arange(-self.Hstart, self.Hstart + self.Hstep, self.Hstep)))

        self.Happ = np.array([np.sin(10.0 * np.pi / 180), 0, np.cos(10.0 * np.pi / 180)])  # Applied field vector
        self.Han = 0.05 * self.ani_coef
        self.Hani = np.array([np.sin(self.thetaAni) * np.cos(self.phiAni), 
                              np.sin(self.thetaAni) * np.sin(self.phiAni), 
                              np.cos(self.thetaAni)])  # Anisotropy axis vector
        self.Hd = 0.0
        self.J = np.array([1, 0, 0])  # Current direction

        # Placeholder for magnetization vectors
        self.M_vectors = np.zeros((len(self.H), 3))

    def setup_output_directory(self):
        # Directory setup for saving results
        event_name = f"test_DEG-{np.degrees(self.phi)}-{np.degrees(self.theta)}_Ani-{self.ani_coef}-{np.degrees(self.phiAni)}-{np.degrees(self.thetaAni)}"
        self.output_dir = f"../Outputs/py/{event_name}/Plots"
        os.makedirs(self.output_dir, exist_ok=True)

    def funZeeman(self, Happ, M):
        # Zeeman energy: E_Z = -H . M (dot product)
        return -np.dot(Happ, M)

    def funAni(self, Han, Hani, M):
        # Anisotropy energy: E_A = 0.5 * H_A * (cos(theta))^2
        return 0.5 * Han * (np.dot(Hani, M))**2

    def funDem(self, Hd, M):
        # Demagnetizing energy: E_D = 0.5 * H_D * Mz^2 (only the z-component matters)
        return 0.5 * Hd * M[2]**2

    def funEner(self, M, H, Happ, Han, Hani, Hd):
        # Total energy calculation
        return self.funZeeman(Happ, M) + self.funAni(Han, Hani, M) + self.funDem(Hd, M)

    def angle_to_vector(self, theta, phi):
        # Converts spherical coordinates (theta, phi) to Cartesian vector
        return np.array([np.sin(theta) * np.cos(phi), 
                         np.sin(theta) * np.sin(phi), 
                         np.cos(theta)])

    def vector_to_angle(self, M):
        # Converts a vector back to spherical coordinates (theta, phi)
        theta = np.arccos(M[2] / np.linalg.norm(M))
        phi = np.arctan2(M[1], M[0])
        return theta, phi

    def angle_energy(self, params, H):
        # Energy minimization in vector form
        theta, phi = params
        M = self.angle_to_vector(theta, phi)
        return self.funEner(M, H, self.Happ, self.Han, self.Hani, self.Hd)

    def minimize_energy(self, n):
        # Minimize the energy for the nth field
        initial_theta, initial_phi = self.vector_to_angle(self.M_vectors[n - 1])
        result = minimize(
            self.angle_energy, 
            [initial_theta, initial_phi],
            args=(self.H[n],),
            bounds=[(0, np.pi), (-np.pi, np.pi)]
        )
        return self.angle_to_vector(result.x[0], result.x[1])

    def calculate_magnetization(self):
        # Set initial magnetization vector based on initial field
        self.M_vectors[0] = self.angle_to_vector(self.Hstart, self.Hstep)
        
        # Loop through all fields and minimize energy
        for n in range(1, len(self.H)):
            self.M_vectors[n] = self.minimize_energy(n)

        return self.M_vectors

    def calculate_hall_effects(self):
        # Calculate Hall effects
        AMR = np.vstack((self.H, self.DeltaR * (np.dot(self.J, self.M_vectors.T))**2 / self.R)).T
        PHE = np.vstack((self.H, self.J[0] * self.DeltaR * self.A * (-self.M_vectors[:, 0]) * (-self.M_vectors[:, 1]))).T
        AHE = np.vstack((self.H, self.J[0] * self.Rahe * np.dot([0, 0, 1], self.M_vectors.T))).T

        totHE = np.vstack((PHE[:, 0], PHE[:, 1] + AHE[:, 1])).T
        return AMR, PHE, AHE, totHE

    def plot_magnetization(self):
        # Plot magnetization components
        plt.figure(figsize=(24, 8))
        plt.subplot(131)
        plt.plot(self.H, self.M_vectors[:, 0], label="M_x")
        plt.legend()
        plt.subplot(132)
        plt.plot(self.H, self.M_vectors[:, 1], label="M_y")
        plt.legend()
        plt.subplot(133)
        plt.plot(self.H, self.M_vectors[:, 2], label="M_z")
        plt.legend()
        plt.suptitle('Magnetization Components')
        plt.savefig(os.path.join(self.output_dir, "MvsH.png"))
        plt.close()

    def plot_hall_effects(self, AMR, PHE, AHE, totHE):
        # Plot all Hall effects
        plt.figure(figsize=(24, 8))
        plt.subplot(131)
        plt.plot(AMR[:, 0], AMR[:, 1], label="AMR")
        plt.legend()
        plt.subplot(132)
        plt.plot(PHE[:, 0], PHE[:, 1], label="PHE")
        plt.legend()
        plt.subplot(133)
        plt.plot(AHE[:, 0], AHE[:, 1], label="AHE")
        plt.legend()
        plt.suptitle('Hall Effects vs. H')
        plt.savefig(os.path.join(self.output_dir, "HallEffects.png"))
        plt.close()

        # Plot total Hall Effect vs H
        plt.figure(figsize=(12, 8))
        plt.plot(totHE[:, 0], totHE[:, 1])
        plt.title('Total Hall Effect vs H')
        plt.savefig(os.path.join(self.output_dir, "totHE.png"))
        plt.close()

    def save_data(self, totHE):
        # Save total Hall effect data to a text file
        event_name = f"test_DEG-{np.degrees(self.phi)}-{np.degrees(self.theta)}_Ani-{self.ani_coef}-{np.degrees(self.phiAni)}-{np.degrees(self.thetaAni)}"
        np.savetxt(f"../Outputs/py/{event_name}/PHE&AHE_{np.degrees(self.phi)}_{np.degrees(self.theta)}_{self.ani_coef}.txt", totHE)


if __name__ == "__main__":
    # User input
    theta = float(input("ThetaH in degrees: "))
    phi = float(input("PhiH in degrees: "))
    ani_coef = float(input("Anisotropy field (coef x 0.05): "))
    thetaAni = float(input("ThetaHan in degrees: "))
    phiAni = float(input("PhiHan in degrees: "))

    # Initialize model
    model = StonerWohlfarthModel(theta, phi, ani_coef, thetaAni, phiAni)
    
    # Calculate magnetization and Hall effects
    M_vectors = model.calculate_magnetization()
    AMR, PHE, AHE, totHE = model.calculate_hall_effects()
    
    # Plot results
    model.plot_magnetization()
    model.plot_hall_effects(AMR, PHE, AHE, totHE)
    
    # Save Hall effect data
    model.save_data(totHE)
