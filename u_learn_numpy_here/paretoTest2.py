
import numpy as np

a, m = 3., 2.  # shape and mode
s = (np.random.pareto(a, 1000) + 1) * m

print(s)

import matplotlib.pyplot as plt
# hist means 直方图 
count, bins, _ = plt.hist(s, 100, density=True)
# then we plot 红色的线
fit = a*m**a / bins**(a+1)
plt.plot(bins, max(count)*fit/max(fit), linewidth=2, color='r')
plt.show()

# see output in ./img/stage2pareto.jpg