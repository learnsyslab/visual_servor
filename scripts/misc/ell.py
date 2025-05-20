import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import serving_demo as sd


rx = 0.75
ry = 0.5
A = np.diag([1.0 / rx**2, 1.0 / ry**2])
c = np.array([0.25, 0])

p = np.array([0.5, 0.1])
n = sd.unit(A @ (p - c))

r = p + n
pn = c + sd.unit(p - c)

plt.figure()
ax = plt.gca()

patch = Ellipse(c, 2*rx, 2*ry, fill=False)
ax.add_patch(patch)
ax.set_aspect("equal")

ax.plot(p[0], p[1], "o", color="red")

ax.plot([p[0], r[0]], [p[1], r[1]], color="green")
ax.plot([c[0], pn[0]], [c[1], pn[1]], color="green")

plt.grid()
plt.xlim([-1, 1])
plt.ylim([-1, 1])

plt.show()
