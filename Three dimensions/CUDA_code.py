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
ke = 60
ic = int(ie / 2)
jc = int(ie / 2)
kc = int(ie / 2)
ex = np.zeros((ie,je,ke))
ey = np.zeros((ie,je,ke))
ez = np.zeros((ie,je,ke))
#dx = np.zeros((ie,je,ke))
#dy = np.zeros((ie,je,ke))
#dz = np.zeros((ie,je,ke))
hx = np.zeros((ie,je,ke))
hy = np.zeros((ie,je,ke))
hz = np.zeros((ie,je,ke))
#gax = np.ones((ie,je,ke))
#gay = np.ones((ie,je,ke))
gaz = np.ones((ie,je,ke))

ddx = 0.01 # Cell size
dt = ddx / 6e8 # Time step size
epsz = 8.854e-12
# Specify the dipole
gaz[ic, jc, kc - 10:kc + 10] = 0
gaz[ic, jc, kc] = 1
# Pulse Parameters
t0 = 20
spread = 6
nsteps = 1

mod = SourceModule("""
    #include <math.h>
    #include <stdio.h>
    #define M_PI 3.14159265358979323846
  __global__ void fdtd_e(double *ex, double *ey, double *ez, double *hx, double *hy, double *hz, double *gaz, int kc, int time_step)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    int tam = kc*2;
	int index = i + tam * (j + tam * k);    
    if (i > 0 && i < 2*kc && j > 0 && j < 2*kc && k > 0 && k < 2*kc){
		ex[index] = ex[index] + 0.5 * (hz[index] - hz[i + tam * ((j - 1) + tam * k)] - hy[index] + hx[i + tam * (j + tam * (k-1))]);
		ey[index] = ey[index] + 0.5 * (hx[index] - hx[i + tam * (j + tam * (k-1))] - hz[index] + hz[(i-1) + tam * (j + tam * k)]);
		ez[index] = ez[index] + 0.5 * gaz[index] * (hy[index] - hy[(i-1) + tam * (j + tam * k)] - hx[index] + hx[i + tam * ((j-1) + tam * k)]);
    }
    if (i == kc && j == kc && k == kc){
		double pulse = exp(pow(-0.5 * ((20 - time_step) / 6),2));	
		ez[index] = pulse;
	}
  }
  
  __global__ void fdtd_h(double *ex, double *ey, double *ez, double *hx, double *hy, double *hz, int kc, int time_step)
  {	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i > 0 && i < 2*kc && j > 0 && j < 2*kc && k > 0 && k < 2*kc){
    	int tam = kc*2;
		int index = i + tam * (j + tam * k);
		hx[index] = hx[index] + 0.5 * (ey[i + tam * (j + tam * (k+1))] - ey[index] - ez[i + tam * ((j+1) + tam * k)] + ez[index]);
		hy[index] = hy[index] + 0.5 * (ez[(i+1) + tam * (j + tam * k)] - ez[index] - ex[i + tam * (j + tam * (k+1))] + ex[index]);
		hz[index] = hz[index] + 0.5 * (ex[i + tam * ((j+1) + tam * k)] - ex[index] - ey[(i+1) + tam * (j + tam * k)] + ey[index]);
	}
  }
  
  
  """)
def func(v):
	func_e = mod.get_function("fdtd_e")
	func_h = mod.get_function("fdtd_h")
	ie = v
	je = v
	ke = v
	nsteps = v
	ic = int(ie / 2)
	jc = int(ie / 2)
	kc = int(ie / 2)
	ex = np.zeros((ie,je,ke))
	ey = np.zeros((ie,je,ke))
	ez = np.zeros((ie,je,ke))
	hx = np.zeros((ie,je,ke))
	hy = np.zeros((ie,je,ke))
	hz = np.zeros((ie,je,ke))
	gaz = np.ones((ie,je,ke))

	ex = ex.astype(np.float64)
	ey = ey.astype(np.float64)
	ez = ez.astype(np.float64)
	hx = hy.astype(np.float64)
	hy = hy.astype(np.float64)
	hz = hz.astype(np.float64)
	gaz = gaz.astype(np.float64)
		
	t = time.time()
	ex_gpu = cuda.mem_alloc(ex.nbytes)
	cuda.memcpy_htod(ex_gpu, ex)
	ey_gpu = cuda.mem_alloc(ey.nbytes)
	cuda.memcpy_htod(ey_gpu, ey)
	ez_gpu = cuda.mem_alloc(ez.nbytes)
	cuda.memcpy_htod(ez_gpu, ez)

	hx_gpu = cuda.mem_alloc(hx.nbytes)
	cuda.memcpy_htod(hx_gpu, hx)
	hy_gpu = cuda.mem_alloc(hy.nbytes)
	cuda.memcpy_htod(hy_gpu, hy)
	hz_gpu = cuda.mem_alloc(hz.nbytes)
	cuda.memcpy_htod(hz_gpu, hz)

	gaz_gpu = cuda.mem_alloc(gaz.nbytes)
	cuda.memcpy_htod(gaz_gpu, gaz)

	
	for time_step in range(1, nsteps + 1):
		func_e(ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, gaz_gpu, np.int32(kc), np.int32(time_step), block=(32,32,1),grid=(128,64,64))
		func_h(ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, np.int32(kc), np.int32(time_step), block=(32,32,1),grid=(128,64,64))
		ez = np.empty_like(ez)
		cuda.memcpy_dtoh(ez, ez_gpu)
		cuda.memcpy_htod(ez_gpu, ez)

	#time.sleep(2)
	#ez = np.empty_like(ez)
	#cuda.memcpy_dtoh(ez, ez_gpu)
	return time.time() - t
for i in tqdm(range(200,400,1)):
	t = func(i)
	arq = open("tempos_cuda_pml",'a')
	arq.writelines(str(i)+" "+str(t)+"\n")
	arq.close()

'''
fig = plt.gcf()
ax = Axes3D(fig)
X = np.arange(0, len(ez), 1)
Z = np.array(ez)
Y = np.arange(0, len(ez), 1)
X, Y = np.meshgrid(X, Y)

ax.plot_surface(X,Y,Z[:,:,kc],cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.view_init(10, 300)
plt.close()
ax.set_zbound(0,0.01)
fig.savefig(str("img-without_cuda.png"),dpi=400)
'''
