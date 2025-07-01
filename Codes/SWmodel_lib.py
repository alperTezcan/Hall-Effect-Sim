import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class SWmodel:
    def __init__(self, ang_or_field=False, verbose=True):
        # initial constant setup
        self.verbose = verbose
        self.setup_constants()
        
        # prepare and angle-sweep or field-sweep experiment
        if ang_or_field: self.setup_angle()
        else: self.setup_mag_field()
        
        # prepare Hd and ThetaJ
        self.setup_Hd_ThetaJ()

        # prepare n-uniaxial anisotropies
        self.setup_uani()

        # prepare cubic anisotropies
        self.setup_cubic()

    def setup_constants(self, R = 1, DeltaR = 0.1, A = 19.36, Rahe = 0.0108):
        self.R = R
        self.DeltaR = DeltaR
        self.A = A
        self.Rahe = Rahe
        if self.verbose: print("constants updated")

    def setup_Hd_ThetaJ(self, Hd = 0.0, ThetaJ = 0.0):
        self.Hd = Hd
        self.ThetaJ = ThetaJ
        if self.verbose: print("Hd and ThetaJ updated")

    def setup_mag_field(self, Hstart = 6, Hstep = 5e-3):
        self.Hstart = Hstart
        self.Hstep = Hstep

        self.H = np.concatenate((np.arange(self.Hstart, -self.Hstart - self.Hstep, -self.Hstep), 
                                 np.arange(-self.Hstart, self.Hstart + self.Hstep, self.Hstep)))
        
        if self.verbose: print("creating field sweep experiment")
        
    def setup_angle(self):
        #if self.verbose: print("creating angle sweep experiment")
        pass
    
    def setup_uani(self, n=0):
        self.n_uani = n
        self.uanis = np.array([[0, 0, 0]])
        
        for i in range(self.n_uani):
            print(f"Info for {i+1}-th anisotropy axis")
            Han = float(input("Anisotropy field (Han): "))
            theta_an = np.radians(float(input("ThetaHan in degrees: ")))
            phi_an = np.radians(float(input("PhiHan in degrees: ")))
            self.uanis = np.append(self.uanis, [[Han, theta_an, phi_an]], axis=0)
        
        if self.verbose: print(f"{self.n_uani} different uniaxial anisotropies added")

    def setup_cubic(self):
        #if self.verbose: print("cubic anisotropy added")
        pass

    def J(self):
        return np.array([np.cos(self.ThetaJ), np.sin(self.ThetaJ), 0])
    
    def funZeeman(self, Happ, ThetaH, PhiH, ThetaM, PhiM):
        # Zeeman energy calculation
        return -Happ * (np.sin(ThetaH) * np.cos(PhiH) * np.sin(ThetaM) * np.cos(PhiM) +
                        np.sin(ThetaH) * np.sin(PhiH) * np.sin(ThetaM) * np.sin(PhiM) +
                        np.cos(ThetaH) * np.cos(ThetaM))
        
    def funAni(self, Han, ThetaAn, PhiAn, ThetaM, PhiM):
        # Uniaxial anisotropy energy calculation
        return 0.5 * Han * (np.sin(
            np.arccos(np.sin(ThetaAn) * np.cos(PhiAn) * np.sin(ThetaM) * np.cos(PhiM) + 
                      np.sin(ThetaAn) * np.sin(PhiAn) * np.sin(ThetaM) * np.sin(PhiM) +
                      np.cos(ThetaAn) * np.cos(ThetaM))))**2
        
    def funDem(self, Hd, ThetaM):
        # Demagnetizing energy
        return 0.5 * Hd * np.cos(ThetaM)**2

    def funEner(self, Happ, ThetaH, PhiH, ThetaM, PhiM, Hd):
        # Total energy calculation
        result = self.funZeeman(Happ, ThetaH, PhiH, ThetaM, PhiM) + self.funDem(Hd, ThetaM)
        for i in range(self.n_uani):
            result += self.funAni(self.uanis[i+1][0], 
                                  self.uanis[i+1][1], 
                                  self.uanis[i+1][2], ThetaM, PhiM)
        return result        
        
    def angle_energy(self, params, H, ThetaH, PhiH):
        # Energy minimization function
        theta, phi = params
        return self.funEner(H, ThetaH, PhiH, theta, phi, self.Hd)

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
        plt.plot(M[:, 0], M[:, 1], linestyle="-", marker="o", markerfacecolor="black", markeredgecolor="black")
        plt.xlabel("H")
        plt.ylabel("Mx")
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        
        plt.subplot(132)
        plt.plot(M[:, 0], M[:, 2], linestyle="-", marker="o", markerfacecolor="black", markeredgecolor="black")
        plt.xlabel("H")
        plt.ylabel("My")
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)        
        
        plt.subplot(133)
        plt.plot(M[:, 0], M[:, 3], linestyle="-", marker="o", markerfacecolor="black", markeredgecolor="black")
        plt.xlabel("H")
        plt.ylabel("Mz")
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        
        plt.suptitle('Magnetization Components')
        plt.show()

    def hall_effects(self, ThetaH, PhiH):
        # Hall effect calculations
        M = self.calculate_hysteresis(ThetaH, PhiH)[:, 1:4]

        AMR = np.vstack((self.H, self.DeltaR * (np.dot(self.J(), M.T))**2 / self.R)).T
        PHE = np.vstack((self.H, self.J()[0] * self.DeltaR * self.A * (-M[:, 0]) * (-M[:, 1]))).T
        AHE = np.vstack((self.H, self.J()[0] * self.Rahe * np.dot([0, 0, 1], M.T))).T

        totHE = np.vstack((PHE[:, 0], PHE[:, 1] + AHE[:, 1])).T
        return (AMR, PHE, AHE, totHE)

    def plot_hall_effects(self, ThetaH_deg, PhiH_deg):
        # Compute hall effects
        ThetaH = np.radians(ThetaH_deg)
        PhiH = np.radians(PhiH_deg)
        
        Halls = self.hall_effects(ThetaH, PhiH)

        ## Plot all Hall effects vs H
        plt.figure(figsize=(24, 8))
        
        plt.subplot(131)
        plt.plot(Halls[0][:, 0], Halls[0][:, 1], linestyle="-", marker="o", label="AMR")
        plt.xlabel("H")
        plt.ylabel("AMR")

        plt.subplot(132)
        plt.plot(Halls[1][:, 0], Halls[1][:, 1], linestyle="-", marker="o", label="PHE")
        plt.xlabel("H")
        plt.ylabel("PHE")

        plt.subplot(133)
        plt.plot(Halls[2][:, 0], Halls[2][:, 1], linestyle="-", marker="o", label="AHE")      
        plt.xlabel("H")
        plt.ylabel("AHE")

        plt.suptitle('Hall Effects vs. H')
        plt.show()

    def plot_totHE(self, ThetaH_deg, PhiH_deg):   
        # Compute hall effects
        ThetaH = np.radians(ThetaH_deg)
        PhiH = np.radians(PhiH_deg)
        
        Halls = self.hall_effects(ThetaH, PhiH)

        # Plot the total HE
        plt.figure(figsize=(12, 8))
        plt.plot(Halls[3][:, 0], Halls[3][:, 1])
        plt.xlabel("H")
        plt.ylabel("AHE")
        plt.title('Total Hall Effect vs H')
        plt.show()

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
            switching_fields = self.calculate_switching_field(thetaH_deg, 0)[:, 0]
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
        plt.xlabel('Hx')
        plt.ylabel('Hy')
        plt.title('Stoner-Wohlfarth Asteroid - THETA')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()

    def plot_asteroid_phi(self, resolution=30):
        phi_vals = np.linspace(0, 360, resolution)
        Hx_, Hy_ = [], []

        for phiH_deg in phi_vals:                  
            switching_fields = self.calculate_switching_field(90.0, phiH_deg)[:, 0]
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
        plt.xlabel('Hx')
        plt.ylabel('Hy')
        plt.title('Stoner-Wohlfarth Asteroid - Phi')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()

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

        ast3D_phi = FuncAnimation(fig, rotate_phi, frames=np.arange(0, 360, 2), interval=100)
        ast3D_theta = FuncAnimation(fig, rotate_theta, frames=np.arange(0, 180, 2), interval=100)

        
        # Display animations inline in Jupyter Notebook
        display(HTML(ast3D_phi.to_jshtml()))  # Show azimuth rotation animation
        display(HTML(ast3D_theta.to_jshtml()))  # Show elevation rotation animation
        
        plt.show()    
