import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Пути к файлам
csv_path = r"C:\Experiments\BEBLaDII\experiments\phase 2\history(in).csv"
output_path = r"C:\Experiments\BEBLaDII\experiments\phase 2\ce_loss_plot.png"

# Чтение данных (разделитель точка с запятой)
df = pd.read_csv(csv_path, sep=';')

# Построение графика
plt.figure(figsize=(14, 8))

# Сглаживание (Гауссовский фильтр)
smoothed = gaussian_filter1d(df['smart-paper-27 - train/ce_loss'], sigma=20)

# Оригинальные данные (полупрозрачно)
plt.plot(df['Step'], df['smart-paper-27 - train/ce_loss'], label='CE Loss (Original)', color='lightblue', alpha=0.7, linewidth=1.0)
# Сглаженные данные (ярко)
plt.plot(df['Step'], smoothed, label='CE Loss (Smoothed)', color='blue', linewidth=2.0)

plt.title('CE Loss over Steps')
plt.xlabel('Step')
plt.ylabel('CE Loss (Log Scale)')
plt.yscale('log')
plt.grid(True, which='major', linestyle='--', alpha=0.7)
plt.grid(True, which='minor', linestyle=':', alpha=0.4)
plt.legend()
plt.tight_layout()

# Сохранение графика
plt.savefig(output_path, dpi=300)
