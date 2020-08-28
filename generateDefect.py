import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tools as tls


X = np.loadtxt('X14_SIM.txt', delimiter=',')
Y = np.loadtxt('Y14_SIM.txt', delimiter=',')
Z = np.loadtxt('Z14_SIM.txt', delimiter=',')


dx_vector = np.zeros( (1,) )
dz_vector = np.zeros( (1,) )

radius_vector = np.zeros( (1,) )
std_radius_vector = np.zeros( (1,) )

maxXYZ, amax = tls.maxpointsXYZ(X, Y, Z)
damagex = 0.05 * ( X[0,-1] - X[0,0] )
damagez = 0.01 * ( np.max(Z[0,:]) - np.min(Z[0,:]) )
yArea1 = 0.4 * Y[:,-1]
yArea2 = 0.6 * Y[:,-1]

dx = X[0,1] - X[0,0]
dy = Y[1,0] - Y[0,0]
dz = Z[0,1] - Z[0,0]

print('X \n', X[0:4, 0:4])
print('Y \n', Y[0:3, 0:3])
print('Z \n', Z[0:3, 0:3])
print(dx, dy, dz)

for j in range(0, len(dx_vector)):
    print('Test')


'''
fig = plt.figure()
ax = fig.gca(projection='3d')
surface = ax.plot_surface(X, Y, Z,  rstride=3, cstride=3, cmap='Greys', linewidth=0.25, vmin = -1.5, vmax = 1)
ax.plot(maxXYZ[0,:],maxXYZ[1,:],maxXYZ[2,:], 'ko')
plt.axis('off')
ax.view_init(-30, 80)
ax.margins(x=0, y=0)
plt.show()
'''