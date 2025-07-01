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

        self.PhiHapp = 10.0 * np.pi / 180
        self.ThetaHapp = 30.0 * np.pi / 180
        self.Han = 0.05 * self.ani_coef
        self.PhiHan = self.phiAni
        self.ThetaHan = self.thetaAni
        self.Hd = 0.0
        self.ThetaJ = 0.0 * np.pi / 180

        # Placeholder for magnetization angles
        self.PHITHETA = np.zeros((len(self.H), 2))
        self.PHITHETA[0] = [self.PhiHapp * np.sign(self.Hstart), self.ThetaHapp * np.sign(self.Hstart)]

    def setup_output_directory(self):
        # Directory setup for saving results
        event_name = f"test_DEG-{np.degrees(self.phi)}-{np.degrees(self.theta)}_Ani-{self.ani_coef}-{np.degrees(self.phiAni)}-{np.degrees(self.thetaAni)}"
        self.output_dir = f"../Outputs/py/{event_name}/Plots"
        os.makedirs(self.output_dir, exist_ok=True)

    def J(self, ThetaJ):
        # Current vector
        return np.array([np.cos(ThetaJ), np.sin(ThetaJ), 0])

    def funZeeman(self, Happ, ThetaH, PhiH, ThetaM, PhiM):
        # Zeeman energy calculation
        return -Happ * (np.sin(ThetaH) * np.cos(PhiH) * np.sin(ThetaM) * np.cos(PhiM) +
                        np.sin(ThetaH) * np.sin(PhiH) * np.sin(ThetaM) * np.sin(PhiM) +
                        np.cos(ThetaH) * np.cos(ThetaM))

    def funAni(self, Han, ThetaAn, PhiAn, ThetaM, PhiM):
        # Anisotropy energy calculation with clipping to avoid numerical issues
        cos_theta = np.clip(np.sin(ThetaAn) * np.cos(PhiAn) * np.sin(ThetaM) * np.cos(PhiM) + 
                            np.sin(ThetaAn) * np.sin(PhiAn) * np.sin(ThetaM) * np.sin(PhiM) +
                            np.cos(ThetaAn) * np.cos(ThetaM), -1, 1)
        return 0.5 * Han * (np.sin(np.arccos(cos_theta)))**2

    def funDem(self, Hd, ThetaM):
        # Demagnetizing energy
        return 0.5 * Hd * np.cos(ThetaM)**2

    def funEner(self, Happ, ThetaH, PhiH, ThetaM, PhiM, Han, ThetaAn, PhiAn, Hd):
        # Total energy calculation
        return (self.funZeeman(Happ, ThetaH, PhiH, ThetaM, PhiM) +
                self.funAni(Han, ThetaAn, PhiAn, ThetaM, PhiM) +
                self.funDem(Hd, ThetaM))

    def angle_energy(self, params, H):
        # Energy minimization function
        phi, theta = params
        return self.funEner(H, self.ThetaHapp, self.PhiHapp, theta, phi, 
                            self.Han, self.ThetaHan, self.PhiHan, self.Hd)

    def minimize_energy(self, n):
        # Minimize the energy for the nth field
        result = minimize(
            self.angle_energy, 
            self.PHITHETA[n - 1],
            args=(self.H[n],),
            bounds=[(-np.pi, np.pi), (0, np.pi)]
        )
        return result.x

    def calculate_magnetization(self):
        # Loop through all fields and minimize energy
        for n in range(1, len(self.H)):
            self.PHITHETA[n] = self.minimize_energy(n)

        listTrackM = np.array(
            [[self.H[n],
              np.sin(self.PHITHETA[n][1]) * np.cos(self.PHITHETA[n][0]),
              np.sin(self.PHITHETA[n][1]) * np.sin(self.PHITHETA[n][0]),
              np.cos(self.PHITHETA[n][1])] for n in range(len(self.H))]
        )
        self.M = listTrackM[:, 1:4]
        return listTrackM

    def calculate_hall_effects(self):
        # Calculate Hall effects
        AMR = np.vstack((self.H, self.DeltaR * (np.dot(self.J(self.ThetaJ), self.M.T))**2 / self.R)).T
        PHE = np.vstack((self.H, self.J(self.ThetaJ)[0] * self.DeltaR * self.A * (-self.M[:, 0]) * (-self.M[:, 1]))).T
        AHE = np.vstack((self.H, self.J(self.ThetaJ)[0] * self.Rahe * np.dot([0, 0, 1], self.M.T))).T

        totHE = np.vstack((PHE[:, 0], PHE[:, 1] + AHE[:, 1])).T
        return AMR, PHE, AHE, totHE

    def plot_magnetization(self, listTrackM):
        # Plot magnetization components
        plt.figure(figsize=(24, 8))
        plt.subplot(131)
        plt.plot(listTrackM[:, 0], listTrackM[:, 1], label="M_x")
        plt.legend()
        plt.subplot(132)
        plt.plot(listTrackM[:, 0], listTrackM[:, 2], label="M_y")
        plt.legend()
        plt.subplot(133)
        plt.plot(listTrackM[:, 0], listTrackM[:, 3], label="M_z")
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
    listTrackM = model.calculate_magnetization()
    AMR, PHE, AHE, totHE = model.calculate_hall_effects()
    
    # Plot results
    model.plot_magnetization(listTrackM)
    model.plot_hall_effects(AMR, PHE, AHE, totHE)
    
    # Save Hall effect data
    model.save_data(totHE)
