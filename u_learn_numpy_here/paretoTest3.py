import numpy as np

from scipy.stats import pareto
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

#Calculate a few first moments:

b = 1# b is the highest point
mean, var, skew, kurt = pareto.stats(b, moments='mvsk')

#Display the probability density function (pdf):

x = np.linspace(pareto.ppf(0.01, b), pareto.ppf(0.99, b), 100)
ax.plot(x, pareto.pdf(x, b, 3), 'r-', lw=5, alpha=0.6, label='pareto pdf')
some = pareto.pdf(4.2, b, 3)
print(some)
plt.show()