import pandas as pd
import matplotlib.pyplot as plt


astar_df = pd.read_csv('run3/astardwa_maze_simulation_results.csv')
custom_df = pd.read_csv('run3/customastar_maze_simulation_results.csv')


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))


ax1.plot(astar_df['Difficulty'], astar_df['Avg Path Length'], marker='o', label='A* DWA')
ax1.plot(custom_df['Difficulty'], custom_df['Avg Path Length'], marker='s', label='Custom A*')
ax1.set_xlabel('Difficulty')
ax1.set_ylabel('Average Path Length')
ax1.set_title('Difficulty vs Average Path Length')
ax1.legend()
ax1.grid(True)

ax2.plot(astar_df['Difficulty'], astar_df['Avg Time'], marker='o', label='A* DWA')
ax2.plot(custom_df['Difficulty'], custom_df['Avg Time'], marker='s', label='Custom A*')
ax2.set_xlabel('Difficulty')
ax2.set_ylabel('Average Time')
ax2.set_title('Difficulty vs Average Time')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('maze_simulation_graphs.png')
plt.show()