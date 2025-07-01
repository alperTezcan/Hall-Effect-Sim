filename = "VAngle_Planar_RT_+1T_+1mA"
data_dir = f"../Data/{filename}.dat"

new_columns = ['Ang. Meas. (°)', 'Ang. Cons. (°))', 'V1xx (V)', 'I1xx(A)', 'R1xx(Ohm)', 
              'V2xy (V)', 'I2xx (A)', 'R2xy (Ohm)', 'Virt', 'Iload ((A)', 'Rvirt (Ohm)',
              'Vload (V)', 'Ivirt', 'Rvirt (Ohm).1', 'Temp (K)', 'Applied Field (G)']

# Read the file
with open(data_dir, "r", encoding='ISO-8859-1') as file:
    lines = file.readlines()

# Replace the header line with the new column names
lines[0] = '\t'.join(new_columns) + '\n'

# Write the updated file back
with open(data_dir, "w", encoding='ISO-8859-1') as file:
    file.writelines(lines)

print("Column names added successfully.")