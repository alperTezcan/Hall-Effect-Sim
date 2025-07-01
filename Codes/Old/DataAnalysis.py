import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set Data directory and Output Directory
filename = "VH_Planar_RT_Rinf_+1mA_45°_3s"
data_dir = f"../Data/{filename}.dat"
output_dir = f"../Outputs/data_py/{filename}"

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(data_dir, delimiter='\t', encoding='ISO-8859-1')

df.columns = df.columns.str.strip()

for col in df.columns:
    df[col] = df[col].str.replace(',', '.')
    df[col] = df[col].astype(float)


"""
# Plot 1: Hall Voltage (V1xx) vs Magnetic Field (Ang. Meas. (°))
plt.figure()
plt.plot(df['Hmes(G)'], df['V1xx (V)'], label='V1xx (V)', color='blue')
plt.xlabel('Magnetic Field')
plt.ylabel('Hall Voltage (V)')
plt.title('Hall Voltage vs Magnetic Field Angle')
plt.legend()
plt.grid(True)
plt.show()


# Plot 2: Hall Resistance (R1xx) vs Magnetic Field (Ang. Meas. (°))
plt.figure()
plt.plot(df['Ang. Meas. (°)'], df['R1xx(Ohm)'], label='R1xx (Ohm)', color='red')
plt.xlabel('Magnetic Field Angle (°)')
plt.ylabel('Hall Resistance (Ohm)')
plt.title('Hall Resistance vs Magnetic Field Angle')
plt.legend()
plt.grid(True)
plt.show()

# Plot 3: Current (I1xx) vs Magnetic Field (Ang. Meas. (°))
plt.figure()
plt.plot(df['Ang. Meas. (°)'], df['I1xx(A)'], label='I1xx (A)', color='green')
plt.xlabel('Magnetic Field Angle (°)')
plt.ylabel('Current (A)')
plt.title('Current vs Magnetic Field Angle')
plt.legend()
plt.grid(True)
plt.show()
"""

"""
# Plot 4: Load Voltage (Vload) vs Magnetic Field (Ang. Meas. (°))
plt.figure()
plt.plot(df['Hmes(G)'], df['Vload (V)'], label='Vload (V)', color='purple')
plt.xlabel('Magnetic Field')
plt.ylabel('Load Voltage (V)')
plt.title('Load Voltage vs Magnetic Field Angle')
plt.legend()
plt.grid(True)
plt.show()



# Plot 5: Hall Resistance (R1xx) vs Load Current (Iload)
plt.figure()
plt.plot(df['Iload ((A)'], df['R1xx(Ohm)'], label='R1xx (Ohm)', color='orange')
plt.xlabel('Load Current (A)')
plt.ylabel('Hall Resistance (Ohm)')
plt.title('Hall Resistance vs Load Current')
plt.legend()
plt.grid(True)
plt.show()
"""

"""
# Plot Hysteresis Loop: Hall Voltage (V1xx) vs Magnetic Field (Ang. Meas. (°))
plt.figure()
plt.plot(df['Ang. Meas. (°)'], df['V1xx (V)'], label='V1xx (V)', color='blue')
plt.xlabel('Magnetic Field Angle (°)')
plt.ylabel('Hall Voltage (V)')
plt.title('Hysteresis Loop: Hall Voltage vs Magnetic Field Angle')
plt.grid(True)
plt.legend()
plt.show()

# Alternatively, if V2xy is also relevant:
plt.figure()
plt.plot(df['Ang. Meas. (°)'], df['V2xy (V)'], label='V2xy (V)', color='red')
plt.xlabel('Magnetic Field Angle (°)')
plt.ylabel('Hall Voltage (V)')
plt.title('Hysteresis Loop: Hall Voltage vs Magnetic Field Angle (V2xy)')
plt.grid(True)
plt.legend()
plt.show()
"""

"""
# Create a figure with 3 subplots arranged horizontally
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# First subplot: M_x component (could be V1xx or another component)
axes[0].plot(df['Ang. Meas. (°)'], df['V1xx (V)'], label='M_x')
axes[0].set_xlabel('Magnetic Field Angle (°)')
axes[0].set_ylabel('Magnetization (Arbitrary Units)')
axes[0].set_title('M_x')
axes[0].legend()
axes[0].grid(True)

# Second subplot: M_y component (could be another relevant data like V2xy)
axes[1].plot(df['Ang. Meas. (°)'], df['V2xy (V)'], label='M_y')
axes[1].set_xlabel('Magnetic Field Angle (°)')
axes[1].set_ylabel('Magnetization (Arbitrary Units)')
axes[1].set_title('M_y')
axes[1].legend()
axes[1].grid(True)

# Third subplot: M_z component (could be a third relevant column)
axes[2].plot(df['Ang. Meas. (°)'], df['R1xx(Ohm)'], label='M_z')
axes[2].set_xlabel('Magnetic Field Angle (°)')
axes[2].set_ylabel('Magnetization (Arbitrary Units)')
axes[2].set_title('M_z')
axes[2].legend()
axes[2].grid(True)

# Set a common title for the whole figure
fig.suptitle('Magnetization Components', fontsize=16)

# Adjust layout to make sure labels/titles don't overlap
plt.tight_layout()

# Show the plot
plt.show()
"""