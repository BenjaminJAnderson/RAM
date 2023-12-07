import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import splev, splprep

pts = np.array([[1651, 1145, 598, 824, 1173, 1541, 1703, 1851],
                [246,  379, 881, 2878, 3319, 3193, 2946, 635]])

tck, u = splprep(pts, u=None, s=0.0, per=1) 
u_new = np.linspace(u.min(), u.max(), 1000)
x_new, y_new = splev(u_new, tck, der=0)

plt.plot(pts[0,:], pts[1,:], 'ro')
plt.plot(x_new, y_new, 'b--')
plt.show()