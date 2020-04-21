import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import math
import matplotlib.pyplot as plt
import time
ke = 1000
ex = numpy.zeros(ke)
hy = numpy.zeros(ke)
ex = ex.astype(numpy.float64)
hy = hy.astype(numpy.float64)
kc = ke//2
t0 = 40
spread = 12
nsteps = 2000

kc = numpy.int32(kc)
t0 = numpy.int32(t0)
spread = numpy.int32(spread)
nsteps = numpy.int32(nsteps)

ex_gpu = cuda.mem_alloc(ex.nbytes)
cuda.memcpy_htod(ex_gpu, ex)


hy_gpu = cuda.mem_alloc(hy.nbytes)
cuda.memcpy_htod(hy_gpu, hy)
#a = numpy.random.randn(4,4)
#a = a.astype(numpy.float32)
#a_gpu = cuda.mem_alloc(a.nbytes)
#cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
    #include <math.h>
  __global__ void fdtd_e(double *ex, double *hy, int kc, int t0, int spread, int step)
  {
    int idx = (blockDim.x*blockIdx.x + threadIdx.x);
    //printf("%d ",gridDim.x);
    double pulse;
    if (idx > 0 && idx < 2*kc){
        ex[idx] = ex[idx] + 0.5* (hy[idx - 1] - hy[idx]);
        if (idx == kc){
                pulse = (double)exp((double)-0.5 * pow((double)(t0 - step)/(double)spread, 2));
                ex[kc] = pulse;
        }
    }
  }
  
  __global__ void fdtd_h(double *ex, double *hy, int kc, int t0, int spread, int step)
  {
    int idx = (blockDim.x*blockIdx.x + threadIdx.x);
    if (idx < 2*kc-1)
        hy[idx] = hy[idx] + 0.5 * (ex[idx] - ex[idx + 1]);
  }
  """)

func_e = mod.get_function("fdtd_e")
func_h = mod.get_function("fdtd_h")
ex_get = numpy.empty_like(ex)
hy_get = numpy.empty_like(hy)
tempos_gpu = []
for k in range(1000,1001):    
    ex = numpy.zeros(k)
    hy = numpy.zeros(k)
    ex = ex.astype(numpy.float64)
    hy = hy.astype(numpy.float64)
    ex_gpu = cuda.mem_alloc(ex.nbytes)
    cuda.memcpy_htod(ex_gpu, ex)
    hy_gpu = cuda.mem_alloc(hy.nbytes)
    cuda.memcpy_htod(hy_gpu, hy)
    kc = numpy.int32(k//2)
    t = time.time()
    for i in range(1,nsteps+1):
        func_e(ex_gpu, hy_gpu,kc,t0,spread,numpy.int32(i), block=(256,1,1),grid=(2048,1))
        #cuda.memcpy_dtoh(ex_get, ex_gpu)
        #ex_get[kc] = math.exp(-0.5 * ((t0 - i) / spread) ** 2)
        #cuda.memcpy_htod(ex_gpu, ex_get)
        func_h(ex_gpu, hy_gpu,kc,t0,spread,numpy.int32(i), block=(256,1,1),grid=(2048,1))
    tempos_gpu.append(time.time()-t)
print('GPU',time.time()-t)
ex_get = numpy.empty_like(ex)
cuda.memcpy_dtoh(ex_get, ex_gpu)
plt.plot(ex_get)

plt.ylim(-1,1)
plt.show()

'''
import numpy as np
from math import exp
from matplotlib import pyplot as plt
ke = 200
ex = np.zeros(ke)
hy = np.zeros(ke)
# Pulse parameters
kc = int(ke / 2)
t0 = 40
spread = 12
nsteps = 100
for time_step in range(1, nsteps + 1):
    # Calculate the Ex field
    for k in range(1, ke):
    ex[k] = ex[k] + 0.5 * (hy[k - 1] - hy[k])
    # Put a Gaussian pulse in the middle
    pulse = exp(-0.5 * ((t0 - time_step) / spread) ** 2)
    ex[kc] = pulse
    # Calculate the Hy field
    for k in range(ke - 1):
    hy[k] = hy[k] + 0.5 * (ex[k] - ex[k + 1])
'''
'''
fig, ax = plt.subplots()
ax.plot(x, tempos_cpu, 'b', label='CPU time')
ax.plot(x, tempos_gpu, 'r', label='GPU time')
ax.set_title("FDTD 1D \n GPU (GTX 940) with a block (32,32,1) Threads vs CPU (I7-5550U) \n All simulations execute 1000 steps")
ax.set_xlabel("Size of grid")
ax.set_ylabel("Time in seconds")
ax.legend()

'''