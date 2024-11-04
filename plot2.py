import pandas as pd
import matplotlib.pyplot as plt

# Read the data
astar_dwa = pd.read_csv('astardwa_dynamic_simulation_results.csv')
custom_astar = pd.read_csv('customastar_dynamic_simulation_results.csv')
dwa = pd.read_csv('dwa_dynamic_simulation_results.csv')

# Clean the 'Avg Time' column in dwa dataframe
dwa['Avg Time'] = dwa['Avg Time'].astype(str).str.replace('s', '').astype(float)

# Create a figure with two subplots side by side
plt.figure(figsize=(15, 6))

# First subplot for Average Time
plt.subplot(1, 2, 1)
plt.plot(astar_dwa['Level'], astar_dwa['Avg Time'], '-o', color='#1f77b4', label='A* DWA(Hybrid)')  # Blue
plt.plot(custom_astar['Level'], custom_astar['Avg Time'], '-o', color='#ff7f0e', label='A*(Global)')  # Orange
plt.plot(dwa['Level'], dwa['Avg Time'], '-o', color='#2ca02c', label='Dynamic Window(local)')  # Green
plt.xlabel('Difficulty Level')
plt.ylabel('Average Time (s)')
plt.title('Difficulty Level vs Average Time')
plt.grid(True)
plt.legend()

# Format y-axis for time plot
plt.gca().set_ylim(bottom=0)
max_time = max(
    astar_dwa['Avg Time'].max(),
    custom_astar['Avg Time'].max(),
    dwa['Avg Time'].max()
)
plt.gca().set_ylim(top=max_time * 1.1)
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

# Second subplot for Average Path Length
plt.subplot(1, 2, 2)
plt.plot(astar_dwa['Level'], astar_dwa['Avg Path Length'], '-o', color='#1f77b4', label='A* DWA(Hybrid)')  # Blue
plt.plot(custom_astar['Level'], custom_astar['Avg Path Length'], '-o', color='#ff7f0e', label='A*(Global)')  # Orange
plt.plot(dwa['Level'], dwa['Avg Path Length'], '-o', color='#2ca02c', label='Dynamic Window(local)')  # Green
plt.xlabel('Difficulty Level')
plt.ylabel('Average Path Length')
plt.title('Difficulty Level vs Average Path Length')
plt.grid(True)
plt.legend()

# Format y-axis for path length plot
plt.gca().set_ylim(bottom=0)
max_length = max(
    astar_dwa['Avg Path Length'].max(),
    custom_astar['Avg Path Length'].max(),
    dwa['Avg Path Length'].max()
)
plt.gca().set_ylim(top=max_length * 1.1)
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig('performance_comparison.png')
plt.show()