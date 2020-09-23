import numpy as np
from math import sin, pi
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d
from matplotlib import cm
import time
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
def func(tam):
	ie = tam
	je = tam
	ic = int(ie / 2)
	jc = int(je / 2)
	ez = np.zeros((ie, je))
	dz = np.zeros((ie, je))
	hx = np.zeros((ie, je))
	hy = np.zeros((ie, je))
	ihx = np.zeros((ie, je))
	ihy = np.zeros((ie, je))
	ddx = 0.01 # Cell size
	dt = ddx / 6e8 # Time step size
	# Create Dielectric Profile
	epsz = 8.854e-12
	# Pulse Parameters
	t0 = 40
	spread = 12
	gaz = np.ones((ie, je))
	# Calculate the PML parameters
	gi2 = np.ones(ie)
	gi3 = np.ones(ie)
	fi1 = np.zeros(ie)
	fi2 = np.ones(ie)
	fi3 = np.ones(ie)
	gj2 = np.ones(ie)
	gj3 = np.ones(ie)
	fj1 = np.zeros(ie)
	fj2 = np.ones(ie)
	fj3 = np.ones(ie)
	npml = 8
	for n in range(npml):
		xnum = npml - n
		xd = npml
		xxn = xnum / xd
		xn = 0.33 * xxn ** 3
		gi2[n] = 1 / (1 + xn)
		gi2[ie - 1 - n] = 1 / (1 + xn)
		gi3[n] = (1 - xn) / (1 + xn)
		gi3[ie - 1 - n] = (1 - xn) / (1 + xn)
		gj2[n] = 1 / (1 + xn)
		gj2[je - 1 - n] = 1 / (1 + xn)
		gj3[n] = (1 - xn) / (1 + xn)
		gj3[je - 1 - n] = (1 - xn) / (1 + xn)
		xxn = (xnum - 0.5) / xd
		xn = 0.33 * xxn ** 3
		fi1[n] = xn
		fi1[ie - 2 - n] = xn
		fi2[n] = 1 / (1 + xn)
		fi2[ie - 2 - n] = 1 / (1 + xn)
		fi3[n] = (1 - xn) / (1 + xn)
		fi3[ie - 2 - n] = (1 - xn) / (1 + xn)
		fj1[n] = xn
		fj1[je - 2 - n] = xn
		fj2[n] = 1 / (1 + xn)
		fj2[je - 2 - n] = 1 / (1 + xn)
		fj3[n] = (1 - xn) / (1 + xn)
		fj3[je - 2 - n] = (1 - xn) / (1 + xn)
	nsteps = tam
	# Dictionary to keep track of desired points for plotting
	plotting_points = [
	{'num_steps': 40, 'data_to_plot': None},
	{'num_steps': nsteps, 'data_to_plot': None},
	]
	# Main FDTD Loop
	tempo = time.time()
	valores = []
	for time_step in tqdm(range(1, nsteps + 1)):
		# Calculate Dz
		for j in range(1, je):
			for i in range(1, ie):
				dz[i, j] = gi3[i] * gj3[j] * dz[i, j] + \
				gi2[i] * gj2[j] * 0.5 * \
				(hy[i, j] - hy[i - 1, j] -
				hx[i, j] + hx[i, j - 1])
		# Put a Gaussian pulse in the middle
		pulse = sin(2 * pi * 1500 * 1e6 * dt * time_step)
		dz[ic, jc] = pulse
		ez = gaz * dz # Calculate the Ez field from Dz
		# Calculate the Hx field
		for j in range(je - 1):
			for i in range(ie - 1):
				curl_e = ez[i, j] - ez[i, j + 1]
				ihx[i, j] = ihx[i, j] + curl_e
				hx[i, j] = fj3[j] * hx[i, j] + fj2[j] * \
				(0.5 * curl_e + fi1[i] * ihx[i, j])
		for j in range(0, je - 1):
			for i in range(0, ie - 1):
				curl_e = ez[i, j] - ez[i + 1, j]
				ihy[i, j] = ihy[i, j] + curl_e
				hy[i, j] = fi3[i] * hy[i, j] - fi2[i] * \
				(0.5 * curl_e + fj1[j] * ihy[i, j])
			# Save data at certain points for later plotting
		valores.append(ez)
	np.save("cpu",np.array(valores))
	return time.time() - tempo
'''
for i in range(500,501,10):
	t = func(i)
	arq = open("tempos_cpu",'a')
	arq.writelines(str(i)+" "+str(t)+"\n")
	arq.close()
'''
func(60)
