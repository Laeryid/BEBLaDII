import csv
import os

csv_path = r"C:\Experiments\BEBLaDII\experiments\phase 4\local_checkpoints\diffusion_trajectory.csv"
temp_csv_path = r"C:\Experiments\BEBLaDII\experiments\phase 4\local_checkpoints\diffusion_trajectory_temp.csv"

with open(csv_path, 'r', encoding='utf-8') as fin, open(temp_csv_path, 'w', encoding='utf-8', newline='') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    
    header = next(reader)
    header.append("RunDate")
    writer.writerow(header)
    
    for row in reader:
        row.append("13.09.2026 10:06:00")
        writer.writerow(row)

os.replace(temp_csv_path, csv_path)
print("CSV updated.")
