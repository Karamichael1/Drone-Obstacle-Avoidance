import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('results2.csv')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))


ax1.plot(df['Speed'], df['CustomA*_Time'], 'r-o', label='Custom A*')
ax1.plot(df['Speed'], df['DWA_Time'], 'g-o', label='DWA')
ax1.plot(df['Speed'], df['AStarDWA_Time'], 'b-o', label='A* + DWA')
ax1.set_xlabel('Speed')
ax1.set_ylabel('Time (s)')
ax1.set_title('Speed vs Time Taken')
ax1.legend()
ax1.grid(True)

ax2.plot(df['Speed'], df['CustomA*_PathLength'], 'r-o', label='Custom A*')
ax2.plot(df['Speed'], df['DWA_PathLength'], 'g-o', label='DWA')
ax2.plot(df['Speed'], df['AStarDWA_PathLength'], 'b-o', label='A* + DWA')
ax2.set_xlabel('Speed')
ax2.set_ylabel('Path Length')
ax2.set_title('Speed vs Path Length')
ax2.legend()
ax2.grid(True)


plt.tight_layout()
plt.savefig('algorithm_performance_graphs2.png', dpi=300, bbox_inches='tight')
plt.close()

print("Graphs have been generated and saved as 'algorithm_performance_graphs2.png'")