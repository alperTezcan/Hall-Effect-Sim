import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from matplotlib.animation import FuncAnimation

class SWmodel:
    def __init__(self, ani_coef, thetaAni, phiAni):
        # Convert degrees to radians
        self.ani_coef = ani_coef
        self.ThetaAni = np.radians(thetaAni)
        self.PhiAni = np.radians(phiAni)
        
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

        self.Han = 0.5 * self.ani_coef
        self.Hd = 0.0
        self.ThetaJ = 0.0

    def J(self):
        return np.array([np.cos(self.ThetaJ), np.sin(self.ThetaJ), 0])

    def setup_output_directory(self):
        # Directory setup for saving results
        event_name = f"Han{self.Han}-PhiAni{phiAni}-ThetaAni{thetaAni}"
        self.output_dir = f"../Outputs/py/v2_1/{event_name}/Plots"
        os.makedirs(self.output_dir, exist_ok=True)

    def funZeeman(self, Happ, ThetaH, PhiH, ThetaM, PhiM):
        # Zeeman energy calculation
        return -Happ * (np.sin(ThetaH) * np.cos(PhiH) * np.sin(ThetaM) * np.cos(PhiM) +
                        np.sin(ThetaH) * np.sin(PhiH) * np.sin(ThetaM) * np.sin(PhiM) +
                        np.cos(ThetaH) * np.cos(ThetaM))

    def funAni(self, Han, ThetaAn, PhiAn, ThetaM, PhiM):
        # Anisotropy energy calculation with clipping to avoid numerical issues
        return 0.5 * Han * (np.sin(
            np.arccos(np.sin(ThetaAn) * np.cos(PhiAn) * np.sin(ThetaM) * np.cos(PhiM) + 
                      np.sin(ThetaAn) * np.sin(PhiAn) * np.sin(ThetaM) * np.sin(PhiM) +
                      np.cos(ThetaAn) * np.cos(ThetaM))))**2

    def funDem(self, Hd, ThetaM):
        # Demagnetizing energy
        return 0.5 * Hd * np.cos(ThetaM)**2

    def funEner(self, Happ, ThetaH, PhiH, ThetaM, PhiM, Han, ThetaAn, PhiAn, Hd):
        # Total energy calculation
        return (self.funZeeman(Happ, ThetaH, PhiH, ThetaM, PhiM) +
                self.funAni(Han, ThetaAn, PhiAn, ThetaM, PhiM) +
                self.funDem(Hd, ThetaM))

    def angle_energy(self, params, H, ThetaH, PhiH):
        # Energy minimization function
        theta, phi = params
        return self.funEner(H, ThetaH, PhiH, theta, phi, self.Han, self.ThetaAni, self.PhiAni, self.Hd)

    def minimize_energy(self, H, ThetaH, PhiH, initial_guess):
        # Minimize the energy for a given field H, ThetaH, and PhiH
        result = minimize(
            self.angle_energy, 
            initial_guess,
            args=(H, ThetaH, PhiH)
        )
        return result.x

    def calculate_hysteresis(self, ThetaH, PhiH):
        # Sweep magnetic field for given applied angles
        M = [] 
        initial_guess = [ThetaH * np.sign(self.Hstart), PhiH * np.sign(self.Hstart)]

        for H in self.H:
            theta_phi = self.minimize_energy(H, ThetaH, PhiH, initial_guess)
            initial_guess = theta_phi

            Mx = np.sin(theta_phi[0]) * np.cos(theta_phi[1])
            My = np.sin(theta_phi[0]) * np.sin(theta_phi[1])
            Mz = np.cos(theta_phi[0])

            M.append([H, Mx, My, Mz])

        M = np.array(M)
        return M

    def plot_hysteresis(self, ThetaH_deg, PhiH_deg):
        # Plot hysteresis loops for given applied angles
        ThetaH = np.radians(ThetaH_deg)
        PhiH = np.radians(PhiH_deg)

        M = self.calculate_hysteresis(ThetaH, PhiH)

        plt.figure(figsize=(24, 8))
        
        plt.subplot(131)
        plt.plot(M[:, 0], M[:, 1], linestyle="-", marker="o", label="M_x")
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.legend()
        
        plt.subplot(132)
        plt.plot(M[:, 0], M[:, 2], linestyle="-", marker="o", label="M_y")
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)        
        plt.legend()
        
        plt.subplot(133)
        plt.plot(M[:, 0], M[:, 3], linestyle="-", marker="o", label="M_z")
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)        
        plt.legend()
        
        plt.suptitle('Magnetization Components')
        plt.savefig(os.path.join(self.output_dir, f"MvsH-ThetaApp{ThetaH_deg}-PhiApp{PhiH_deg}.png"))
        plt.close()

    def hall_effects(self, ThetaH, PhiH):
        # Hall effect calculations
        M = self.calculate_hysteresis(ThetaH, PhiH)[:, 1:4]

        AMR = np.vstack((self.H, self.DeltaR * (np.dot(self.J(), M.T))**2 / self.R)).T
        PHE = np.vstack((self.H, self.J()[0] * self.DeltaR * self.A * (-M[:, 0]) * (-M[:, 1]))).T
        AHE = np.vstack((self.H, self.J()[0] * self.Rahe * np.dot([0, 0, 1], M.T))).T

        totHE = np.vstack((PHE[:, 0], PHE[:, 1] + AHE[:, 1])).T
        return (AMR, PHE, AHE, totHE)

    def plot_hall_effects(self, ThetaH_deg, PhiH_deg):
        # Plot hall effects
        ThetaH = np.radians(ThetaH_deg)
        PhiH = np.radians(PhiH_deg)
        
        Halls = self.hall_effects(ThetaH, PhiH)

        ## Plot all Hall effects vs H
        plt.figure(figsize=(24, 8))
        
        plt.subplot(131)
        plt.plot(Halls[0][:, 0], Halls[0][:, 1], linestyle="-", marker="o", label="AMR")
        plt.legend()

        plt.subplot(132)
        plt.plot(Halls[1][:, 0], Halls[1][:, 1], linestyle="-", marker="o", label="PHE")
        plt.legend()

        plt.subplot(133)
        plt.plot(Halls[2][:, 0], Halls[2][:, 1], linestyle="-", marker="o", label="AHE")      
        plt.legend()

        plt.suptitle('Hall Effects vs. H')
        plt.savefig(os.path.join(self.output_dir, f"HallEffects-ThetaApp{ThetaH_deg}-PhiApp{PhiH_deg}.png"))
        plt.close()

        ## Plot total Hall Effect vs H
        plt.figure(figsize=(12, 8))
        plt.plot(Halls[3][:, 0], Halls[3][:, 1])
        plt.title('Total Hall Effect vs H')
        plt.savefig(os.path.join(self.output_dir, f"totHE-ThetaApp{ThetaH_deg}-PhiApp{PhiH_deg}.png"))
        plt.close()

    def calculate_switching_field(self, ThetaH_deg, PhiH_deg, threshold = 0.1):
        ThetaH = np.radians(ThetaH_deg)
        PhiH = np.radians(PhiH_deg)

        M = self.calculate_hysteresis(ThetaH, PhiH)
        dMz = np.diff(M[:, 3])

        switch_indices = np.where(np.abs(dMz) > threshold)[0]
        switching_fields = M[switch_indices]

        return switching_fields 

    def plot_asteroid_theta(self, resolution=30):
        theta_vals = np.linspace(0, 180, resolution)
        Hx_, Hy_ = [], []

        for thetaH_deg in theta_vals:         
            switching_fields = self.calculate_switching_field(thetaH_deg, self.PhiAni)[:, 0]
            if switching_fields.size > 0:
                Hx_.append(switching_fields * np.cos(np.radians(thetaH_deg)))
                Hy_.append(switching_fields * np.sin(np.radians(thetaH_deg)))
        Hx = np.concatenate(Hx_)
        Hy = np.concatenate(Hy_)

        # Plot the astroid
        plt.figure(figsize=(10, 10))
        plt.scatter(Hx, Hy, label="Stoner-Wohlfarth Asteroid - THETA")
        plt.axhline(0, color='black',linewidth=0.5)
        plt.axvline(0, color='black',linewidth=0.5)
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.xlabel(r'$H_x$')
        plt.ylabel(r'$H_y$')
        plt.title('Stoner-Wohlfarth Asteroid - THETA')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"asteroid_theta.png"))
        plt.close()       

    def plot_asteroid_phi(self, resolution=30):
        phi_vals = np.linspace(0, 360, resolution)
        Hx_, Hy_ = [], []

        for phiH_deg in phi_vals:                  
            switching_fields = self.calculate_switching_field(self.ThetaAni, phiH_deg)[:, 0]
            if switching_fields.size > 0:
                Hx_.append(switching_fields * np.cos(np.radians(phiH_deg)))   ## in reality,they are scaled by sin(theta)
                Hy_.append(switching_fields * np.sin(np.radians(phiH_deg)))   ## in reality,they are scaled by sin(theta)
        Hx = np.concatenate(Hx_)
        Hy = np.concatenate(Hy_)

        # Plot the astroid
        plt.figure(figsize=(10, 10))
        plt.scatter(Hx, Hy, label="Stoner-Wohlfarth Asteroid - PHI")
        plt.axhline(0, color='black',linewidth=0.5)
        plt.axvline(0, color='black',linewidth=0.5)
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.xlabel(r'$H_x$')
        plt.ylabel(r'$H_y$')
        plt.title('Stoner-Wohlfarth Asteroid - Phi')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"asteroid_phi.png"))
        plt.close()             


    def plot_asteroid_3d(self, resolution=30):
        theta_vals = np.linspace(0, 180, resolution)
        phi_vals = np.linspace(0, 360, resolution)
        Hx_, Hy_, Hz_ = [], [], []

        # Calculate switching fields for each combination of ThetaH and PhiH
        for thetaH_deg in theta_vals:
            for phiH_deg in phi_vals:
                switching_fields = self.calculate_switching_field(thetaH_deg, phiH_deg)[:, 0]
                if switching_fields.size > 0:
                    Hx_.append(switching_fields * np.cos(np.radians(thetaH_deg)) * np.cos(np.radians(phiH_deg)))
                    Hy_.append(switching_fields * np.cos(np.radians(thetaH_deg)) * np.sin(np.radians(phiH_deg)))
                    Hz_.append(switching_fields * np.sin(np.radians(thetaH_deg)))  
                    print("written")
        Hx = np.concatenate(Hx_)
        Hy = np.concatenate(Hy_)
        Hz = np.concatenate(Hz_)

        points_2d = np.vstack((Hx, Hy)).T
        tri = Delaunay(points_2d)
        
        # Plot the asteroid
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Hx, Hy, Hz, color="red")
        ax.plot_trisurf(Hx, Hy, Hz, 
                        triangles=tri.simplices, 
                        cmap='viridis', edgecolor='none')
        ax.set_xlabel('Hx')
        ax.set_ylabel('Hy')
        ax.set_zlabel('Hz')

        def rotate_phi(angle):
            ax.view_init(elev=30, azim=angle)

        def rotate_theta(angle):
            ax.view_init(elev=angle, azim=35)

        ani_phi = FuncAnimation(fig, rotate_phi, frames=np.arange(0, 360, 2), interval=100)
        ani_theta = FuncAnimation(fig, rotate_theta, frames=np.arange(0, 180, 2), interval=100)

        ani_phi.save(os.path.join(self.output_dir,'3d_rotation_phi.mp4'), writer='ffmpeg')
        ani_theta.save(os.path.join(self.output_dir,'3d_rotation_theta.mp4'), writer='ffmpeg')
        
        plt.savefig(os.path.join(self.output_dir, f"asteroid_3d.png"))
        plt.close()

if __name__ == "__main__":
    # User input for anisotropy
    ani_coef = float(input("Anisotropy field (coef x 0.5): "))
    thetaAni = float(input("ThetaHan in degrees: "))
    phiAni = float(input("PhiHan in degrees: "))

    # Initialize model
    model = SWmodel(ani_coef, thetaAni, phiAni)
    print("Model created for given anisotropy")

    print("############# Plot hysteresis loops for specific angles ##################")
    ThetaH_deg = float(input("ThetaH in degrees for hysteresis loop: "))
    PhiH_deg = float(input("PhiH in degrees for hysteresis loop: "))
    model.plot_hysteresis(ThetaH_deg, PhiH_deg)
    print("Hysteresis loop plotted")
    model.plot_hall_effects(ThetaH_deg, PhiH_deg)
    print("Hall effects are calculated and plotted")

    
    print("############# Plot the Stoner-Wohlfarth asteroid ########################")
    model.plot_asteroid_theta()
    print("Phi is set to PhiAni, Theta is sweeped and Hz vs. H(xy) plotted")
    model.plot_asteroid_phi()
    print("Theta is set to ThetaAni, Phi is sweeped and Hx vs. Hy plotted")
    model.plot_asteroid_3d()
    print("Nothing is set, (Theta, Phi) is sweeped and Hx vs. Hy vs. Hz plotted")
