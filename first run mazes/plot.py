import pandas as pd
import matplotlib.pyplot as plt


astar_df = pd.read_csv('astardwa_dynamic_simulation_results.csv')
custom_df = pd.read_csv('customastar_dynamic_simulation_results.csv')
dynamic_df=pd.read_csv('dwa_dynamic_simulation_results.csv')

# Re-plot with normal scale for y-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot Level vs Average Path Length
ax1.plot(astar_df['Level']*2, astar_df['Avg Path Length'], marker='o', label='A* DWA')
ax1.plot(custom_df['Level']*2, custom_df['Avg Path Length'], marker='s', label='A*(Global)')
ax1.plot(dynamic_df['Level']*2, dynamic_df['Avg Path Length'], marker='s', label='Dynamic Window(local)')
ax1.set_xlabel('Number of Obstacles')
ax1.set_ylabel('Average Path Length')
ax1.set_title('Numebr of obstacles vs Average Path Length')
ax1.legend()
ax1.grid(True)

# Plot Level vs Average Time with normal y-axis scale
ax2.plot(astar_df['Level']*2, astar_df['Avg Time'], marker='o', label='A* DWA(My implementation)')
ax2.plot(custom_df['Level']*2, custom_df['Avg Time'], marker='s', label='A*(Global)')
ax2.plot(dynamic_df['Level']*2, dynamic_df['Avg Time'], marker='s', label='Dynamic Window(local)')
ax2.set_xlabel('Number of obstacles')
ax2.set_ylabel('Average Time(S)')
ax2.set_title('Number of obstacles vs Average Time')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

