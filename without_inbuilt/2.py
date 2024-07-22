# Visualize the n-dimensional data using contour plots.
# Write a program to implement the A* algorithm

# (a)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
Z = np.sin(X * np.pi) * np.cos(Y * np.pi)
contour = plt.contour(X, Y, Z, 20, cmap='viridis')
plt.colorbar(contour)

plt.tight_layout()
plt.show()

# (b)
def h(n):
    H = {'A': 3, 'B': 4, 'C': 2, 'D': 6, 'G': 0, 'S': 5}
    return H[n]

def a_star_algorithm(graph, start, goal):
    open_list = [start]
    closed_list = set()
    g = {start:0}
    parents = {start:start}
    while open_list:
        open_list.sort(key=lambda v: g[v] + h(v), reverse=True)
        n = open_list.pop()
        if n == goal:
            reconst_path = []
            while parents[n] != n:
                reconst_path.append(n)
                n = parents[n]
            reconst_path.append(start)
            reconst_path.reverse()
            print(f'Path found: {reconst_path}')
            return reconst_path
        for (m, weight) in graph[n]:
            if m not in open_list and m not in closed_list:
                open_list.append(m)
                parents[m] = n
                g[m] = g[n] + weight
            else:
                if g[m] > g[n] + weight:
                    g[m] = g[n] + weight
                    parents[m] = n
                    if m in closed_list:
                        closed_list.remove(m)
                        open_list.append(m)
        closed_list.add(n)
    print('Path does not exist!')
    return None


graph = {
    'S': [('A', 1), ('G', 10)],
    'A': [('B', 2), ('C', 1)],
    'B': [('D', 5)],
    'C': [('D', 3),('G', 4)],
    'D': [('G', 2)]
}
a_star_algorithm(graph, 'S', 'G')

