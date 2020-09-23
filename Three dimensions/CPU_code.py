import numpy as np
from math import exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import cm
from tqdm import tqdm
import time
ie = 60
je = 60
ke = 60
ic = int(ie / 2)
jc = int(ie / 2)
kc = int(ie / 2)
ex = np.zeros((ie,je,ke))
ey = np.zeros((ie,je,ke))
ez = np.zeros((ie,je,ke))
dx = np.zeros((ie,je,ke))
dy = np.zeros((ie,je,ke))
dz = np.zeros((ie,je,ke))
hx = np.zeros((ie,je,ke))
hy = np.zeros((ie,je,ke))
hz = np.zeros((ie,je,ke))
gax = np.ones((ie,je,ke))
gay = np.ones((ie,je,ke))
gaz = np.ones((ie,je,ke))

ddx = 0.01 # Cell size
dt = ddx / 6e8 # Time step size
epsz = 8.854e-12
# Specify the dipole
gaz[ic, jc, kc - 10:kc + 10] = 0
gaz[ic, jc, kc] = 0
# Pulse Parameters
t0 = 20
spread = 6
nsteps = 60


def calculate_d_fields(ie, je, ke, dx, dy, dz, hx, hy, hz):
	for i in range(1, ie):
		for j in range(1, je):
			for k in range(1, ke):
				dx[i, j, k] = dx[i, j, k] + 0.5 * (hz[i, j, k] - hz[i, j - 1, k] - hy[i, j, k] + hy[i, j, k - 1])
	for i in range(1, ie):
		for j in range(1, je):
			for k in range(1, ke):
				dy[i, j, k] = dy[i, j, k] + 0.5 * (hx[i, j, k] - hx[i, j, k - 1] - hz[i, j, k] + hz[i - 1, j, k])
	for i in range(1, ie):
		for j in range(1, je):
			for k in range(1, ke):
				dz[i, j, k] = dz[i, j, k] + 0.5 * (hy[i, j, k] - hy[i - 1, j, k] - hx[i, j, k] + hx[i, j - 1, k])
	return dx, dy, dz


def calculate_e_fields(ie, je, ke, dx, dy, dz, gax, gay, gaz, ex, ey, ez):
	for i in range(0, ie):
		for j in range(0, je):
			for k in range(0, ke):
				ex[i, j, k] = gax[i, j, k] * dx[i, j, k]
				ey[i, j, k] = gay[i, j, k] * dy[i, j, k]
				ez[i, j, k] = gaz[i, j, k] * dz[i, j, k]
	return ex, ey, ez

def calculate_h_fields(ie, je, ke, hx, hy, hz, ex, ey, ez):
	for i in range(0, ie):
		for j in range(0, je - 1):
			for k in range(0, ke - 1):
				hx[i, j, k] = hx[i, j, k] + 0.5 * (ey[i, j, k + 1] - ey[i, j, k] - ez[i, j + 1, k] + ez[i, j, k])
	for i in range(0, ie - 1):
		for j in range(0, je):
			for k in range(0, ke - 1):
				hy[i, j, k] = hy[i, j, k] + 0.5 * (ez[i + 1, j, k] - ez[i, j, k] - ex[i, j, k + 1] + ex[i, j, k])
	for i in range(0, ie - 1):
		for j in range(0, je - 1):
			for k in range(0, ke):
				hz[i, j, k] = hz[i, j, k] + 0.5 * (ex[i, j + 1, k] - ex[i, j, k] - ey[i + 1, j, k] + ey[i, j, k])
	return hx, hy, hz
def func(v):
	ie = v
	je = v
	ke = v
	ic = int(ie / 2)
	jc = int(ie / 2)
	kc = int(ie / 2)
	nsteps = v
	ex = np.zeros((ie,je,ke))
	ey = np.zeros((ie,je,ke))
	ez = np.zeros((ie,je,ke))
	dx = np.zeros((ie,je,ke))
	dy = np.zeros((ie,je,ke))
	dz = np.zeros((ie,je,ke))
	hx = np.zeros((ie,je,ke))
	hy = np.zeros((ie,je,ke))
	hz = np.zeros((ie,je,ke))
	gax = np.ones((ie,je,ke))
	gay = np.ones((ie,je,ke))
	gaz = np.ones((ie,je,ke))

	ddx = 0.01 # Cell size
	dt = ddx / 6e8 # Time step size
	epsz = 8.854e-12
	# Specify the dipole
	gaz[ic, jc, kc - 10:kc + 10] = 0
	gaz[ic, jc, kc] = 1
	t = time.time()
	values = []
	for time_step in tqdm(range(1, nsteps + 1)):

		dx, dy, dz = calculate_d_fields(ie, je, ke, dx, dy, dz, hx, hy, hz)

		pulse = exp(-0.5 * ((t0 - time_step) / spread) ** 2)
		
		dz[ic, jc, kc] = pulse
		ex, ey, ez = calculate_e_fields(ie, je, ke, dx, dy, dz, gax, gay, gaz, ex, ey, ez)
		values.append(ez)
		hx, hy, hz = calculate_h_fields(ie, je, ke, hx, hy, hz, ex, ey, ez)
	return ez, np.array(values)
'''
for i in tqdm(range(10,100,1)):
	t = func(i)
	arq = open("tempos_cuda_pml",'a')
	arq.writelines(str(i)+" "+str(t)+"\n")
	arq.close()
'''
ez, v = func(50)
np.save("cpu.npy",v)

fig = plt.gcf()
ax = Axes3D(fig)
X = np.arange(0, len(ez), 1)
Z = np.array(ez)
Y = np.arange(0, len(ez), 1)
X, Y = np.meshgrid(X, Y)

ax.plot_surface(X,Y,Z[:,:,25],cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.view_init(90, 90)
plt.close()
ax.set_zbound(0,0.01)
fig.savefig(str("img-without_cuda.png"),dpi=400)
