import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy as np
from math import sin, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import cm
from tqdm import tqdm
import time
import keras
ie = 60
je = 60
ic = int(ie / 2 - 5)
jc = int(je / 2 - 5)
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



mod = SourceModule("""
    #include <math.h>
    #include <stdio.h>
    #define M_PI 3.14159265358979323846
  __global__ void fdtd_e(double *ez, double *hx, double *hy, int kc, int time_step)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= 0 && i < 2*kc && j > 0 && j < 2*kc){
		ez[i*(kc*2)+j] = ez[i*(kc*2)+j] + 0.5 * ((hy[(i*(kc*2))+j] - hy[((i-1)*(kc*2))+j] - hx[(i*(kc*2))+j] + hx[(i*(kc*2))+(j-1)]));
        
    }
    if (i == kc && j == kc){
		double pulse = sin(2 * M_PI * 1500 * 1e6 * (0.01/(6e8)) * time_step);	
		ez[kc*(kc*2)+kc] = pulse;
	}
  }
  
  __global__ void fdtd_h(double *ez, double *hx, double *hy, int kc)
  {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i > 0 && i < (2*kc)-2 && j > 0 && j < (2*kc)-2){
    	hx[(i*(kc*2))+j] = hx[(i*(kc*2))+j] + (0.5 * (ez[(i*(kc*2))+j] - ez[(i*(kc*2))+(j+1)]));
    	
		hy[(i*(kc*2))+j] = hy[(i*(kc*2))+j] + (0.5 * (ez[((i+1)*(kc*2))+j] - ez[(i*(kc*2))+j] ));
	}
  }
  
  
  """)
def func(k):
	tempo = time.time()
	ez = np.zeros((k,k))
	hx = np.zeros((k,k))
	hy = np.zeros((k,k))

	ez = ez.astype(np.float64)
	hx = hy.astype(np.float64)
	hy = hy.astype(np.float64)

	ez_gpu = cuda.mem_alloc(ez.nbytes)
	cuda.memcpy_htod(ez_gpu, ez)
	#ez_gpu = gpuarray.to_gpu(ez)

	hx_gpu = cuda.mem_alloc(hx.nbytes)
	cuda.memcpy_htod(hx_gpu, hx)
	#hx_gpu = gpuarray.to_gpu(hx)


	hy_gpu = cuda.mem_alloc(hy.nbytes)
	cuda.memcpy_htod(hy_gpu, hy)
	#hy_gpu = gpuarray.to_gpu(hy)

	model = keras.models.load_model('pml-IA')


	kc = np.int32(k//2)
	func_e = mod.get_function("fdtd_e")
	func_h = mod.get_function("fdtd_h")

	nsteps = k
	valores = []
	for time_step in tqdm(range(1, nsteps + 1)):
		
		func_e(ez_gpu, hx_gpu, hy_gpu, kc, np.int32(time_step), block=(32,32,1),grid=(256,256))
		
		func_h(ez_gpu, hx_gpu, hy_gpu, kc, block=(32,32,1),grid=(256,256))

		ez = np.empty_like(ez)
		cuda.memcpy_dtoh(ez, ez_gpu)

		hx = np.empty_like(hx)
		cuda.memcpy_dtoh(hx, hx_gpu)

		hy = np.empty_like(ez)
		cuda.memcpy_dtoh(hy, hy_gpu)
		valores.append(ez)
		for cmd in range(10):
			ez[:,cmd] = np.squeeze(model.predict(ez[:,cmd]))
			ez[:,-1] = np.squeeze(model.predict(ez[:,len(ez)-cmd-1]))
			ez[cmd,:] = np.squeeze(model.predict(ez[cmd,:]))
			ez[len(ez)-cmd-1,:] = np.squeeze(model.predict(ez[len(ez)-cmd-1,:]))

			hx[:,cmd] = np.squeeze(model.predict(hx[:,cmd]))
			hx[:,len(ez)-cmd-1] = np.squeeze(model.predict(hx[:,len(ez)-cmd-1]))
			hx[cmd,:] = np.squeeze(model.predict(hx[cmd,:]))
			hx[len(ez)-cmd-1,:] = np.squeeze(model.predict(hx[len(ez)-cmd-1,:]))
			
			hy[:,cmd] = np.squeeze(model.predict(hy[:,0]))
			hy[:,len(ez)-cmd-1] = np.squeeze(model.predict(hy[:,-1]))
			hy[cmd,:] = np.squeeze(model.predict(hy[cmd,:]))
			hy[len(ez)-cmd-1,:] = np.squeeze(model.predict(hy[len(ez)-cmd-1,:]))

		cuda.memcpy_htod(ez_gpu, ez)
		#cuda.memcpy_htod(hx_gpu, hx)
		#cuda.memcpy_htod(hy_gpu, hy)
	
		
	#weights = model.get_weights()
	#print(weights)
	np.save("cuda_pml",np.array(valores))
	return time.time() - tempo
	'''
	fig = plt.gcf()
	ax = Axes3D(fig)
	X = np.arange(0, len(ez), 1)
	Z = np.array(ez)
	Y = np.arange(0, len(ez), 1)
	X, Y = np.meshgrid(X, Y)

	ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,linewidth=0, antialiased=False)
	ax.view_init(10, 300)
	plt.close()
	ax.set_zbound(-1.2, 1.2)
	fig.savefig(str("img-without_cuda.png"),dpi=400)

for i in tqdm(range(50,501,10)):
	t = func(i)
	arq = open("tempos_cuda_pml",'a')
	arq.writelines(str(i)+" "+str(t)+"\n")
	arq.close()
'''
func(60)
