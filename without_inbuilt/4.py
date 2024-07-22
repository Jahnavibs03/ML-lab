# Visualize the n-dimensional data using Box-plot.
# Write a program to implement Alpha-beta pruning algorithm.

# (a)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = np.random.rand(100, 3)  
df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2', 'Feature 3'])
sns.boxplot(data=df)

plt.tight_layout()
plt.show()

# (b)
MAX, MIN = 1000, -1000

def alphabeta_minimax(depth, nodeIndex, maximizingPlayer, values, alpha, beta):

	if depth == 3:
		return values[nodeIndex]

	if maximizingPlayer:
		best = MIN
		for i in range(0, 2):
			val = alphabeta_minimax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta)
			best = max(best, val)
			alpha = max(alpha, best)

			if beta <= alpha:
				break
		return best

	else:
		best = MAX
		for i in range(0, 2):
			val = alphabeta_minimax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta)
			best = min(best, val)
			beta = min(beta, best)

			if beta <= alpha:
				break

		return best


values = [3, 5, 6, 9, 1, 2, 0, -1]
print("The optimal value is :", alphabeta_minimax(0, 0, True, values, MIN, MAX))

