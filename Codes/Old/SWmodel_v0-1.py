import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

theta = float(input("ThetaH in degrees: "))
phi = float(input("PhiH in degrees: "))
ani_coef = float(input("Anisotropy field (coef x 0.05): "))
thetaAni = float(input("ThetaHan in degrees: "))
phiAni = float(input("PhiHan in degrees: "))

# Directory Setup
event_name = f"test_DEG-{phi}-{theta}_Ani-{ani_coef}-{phiAni}-{thetaAni}"
output_dir = f"../Outputs/py/{event_name}/Plots"
os.makedirs(output_dir, exist_ok=True)

############################### CONSTANTS ##########################################
## Sampling
R = 1
DeltaR = 0.1
A = 19.36
Rahe = 0.0108

## Applied Fields
Hstart = 6
Hstep = 5 / 1000
H = np.concatenate((np.arange(Hstart, -Hstart - Hstep, -Hstep), 
                    np.arange(-Hstart, Hstart + Hstep, Hstep)))

PhiHapp = 10.0 * np.pi / 180
ThetaHapp = 30.0 * np.pi / 180

## Anisotropy
Han = 0.05 * ani_coef
PhiHan = phiAni * np.pi / 180
ThetaHan = thetaAni * np.pi / 180

## Demagnatizing Field
Hd = 0.0

## Current
ThetaJ = 0.0 * np.pi / 180

def J(ThetaJ):
    return np.array([np.cos(ThetaJ), np.sin(ThetaJ), 0])

############################### ENERGY FUNCTIONS ##################################
def funZeeman(Happ, ThetaH, PhiH, ThetaM, PhiM):
    return -Happ * (np.sin(ThetaH) * np.cos(PhiH) * np.sin(ThetaM) * np.cos(PhiM) +
                    np.sin(ThetaH) * np.sin(PhiH) * np.sin(ThetaM) * np.sin(PhiM) +
                    np.cos(ThetaH) * np.cos(ThetaM))

def funAni(Han, ThetaAn, PhiAn, ThetaM, PhiM):
    return 0.5 * Han * (np.sin(
        np.arccos(np.sin(ThetaAn) * np.cos(PhiAn) * np.sin(ThetaM) * np.cos(PhiM) + 
                  np.sin(ThetaAn) * np.sin(PhiAn) * np.sin(ThetaM) * np.sin(PhiM) +
                  np.cos(ThetaAn) * np.cos(ThetaM))))**2

def funDem(Hd, ThetaM):
    return 0.5 * Hd * np.cos(ThetaM)**2

def funEner(Happ, ThetaH, PhiH, ThetaM, PhiM, Han, ThetaAn, PhiAn, Hd):
    return (funZeeman(Happ, ThetaH, PhiH, ThetaM, PhiM) +
            funAni(Han, ThetaAn, PhiAn, ThetaM, PhiM) +
            funDem(Hd, ThetaM))

def angle_energy(params, H, theta_happ, phi_happ, Han, theta_han, phi_han, Hd):
    phi, theta = params
    return funEner(H, theta_happ, phi_happ, theta, phi, Han, theta_han, phi_han, Hd)

################################## MINIMIZER #######################################
PHITHETA = np.zeros((len(H), 2))
PHITHETA[0] = [PhiHapp * np.sign(Hstart), ThetaHapp * np.sign(Hstart)]

def minimize_energy(n, cache = PHITHETA):    
    result = minimize(
        angle_energy, 
        cache[n - 1],
        args=(H[n], ThetaHapp, PhiHapp, Han, ThetaHan, PhiHan, Hd),
        #bounds=[(-np.pi, np.pi), (0, np.pi)]
    )
    return result.x

for n in range(1, len(H)):
    PHITHETA[n] = minimize_energy(n)

listTrackM = np.array(
    [[H[n],
      np.sin(PHITHETA[n][1]) * np.cos(PHITHETA[n][0]),
      np.sin(PHITHETA[n][1]) * np.sin(PHITHETA[n][0]),
      np.cos(PHITHETA[n][1])] for n in range(len(H))]
    )

M = listTrackM[:, 1:4]

################################# HALL EFFECTS ######################################
AMR = np.vstack((H, DeltaR * (np.dot(J(ThetaJ), M.T))**2 / R)).T
PHE = np.vstack((H, J(ThetaJ)[0] * DeltaR * A * (-M[:, 0]) * (-M[:, 1]))).T
AHE = np.vstack((H, J(ThetaJ)[0] * Rahe * np.dot([0, 0, 1], M.T))).T

totHE = np.vstack((PHE[:, 0], PHE[:, 1] + AHE[:, 1])).T

##################################### PLOTS ########################################
## Plot Mx, My, Mz vs H
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
plt.savefig(os.path.join(output_dir, "MvsH.png"))
plt.close()

## Plot all Hall effects vs H
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
plt.savefig(os.path.join(output_dir, "HallEffects.png"))
plt.close()

## Plot total Hall Effect vs H
plt.figure(figsize=(12, 8))
plt.plot(totHE[:, 0], totHE[:, 1])
plt.title('Total Hall Effect vs H')
plt.savefig(os.path.join(output_dir, "totHE.png"))
plt.close()

## Save total Hall Effect data
np.savetxt(f"../Outputs/py/{event_name}/PHE&AHE_{phi}_{theta}_{ani_coef}.txt", totHE)