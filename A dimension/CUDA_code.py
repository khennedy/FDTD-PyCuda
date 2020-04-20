import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import matplotlib.pyplot as plt
import time
ke = 100
ex = numpy.zeros(ke)
hy = numpy.zeros(ke)
ex = ex.astype(numpy.float32)
hy = hy.astype(numpy.float32)
kc = ke//2
t0 = 40
spread = 12
nsteps = 100

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
  __global__ void fdtd(float *ex, float *hy, int kc, int t0, int spread, int step)
  {
    int idx = threadIdx.x + threadIdx.y;
    float pulse;
    ex[idx] = ex[idx] + 0.5* (hy[idx-1] - hy[idx]);
    pulse = (float)exp((float)-0.5 * pow((float)(t0 - step)/(float)spread, 2));
    ex[kc] = pulse;
    idx = threadIdx.x + threadIdx.y;
    hy[idx] = hy[idx] + 0.5 * (ex[idx] - ex[idx+1]);
  }
  """)

func = mod.get_function("fdtd")
ex_get = numpy.empty_like(ex)
hy_get = numpy.empty_like(hy)
t = time.time()
for i in range(1,nsteps+1):
    func(ex_gpu, hy_gpu,kc,t0,spread,numpy.int32(i), block=(32,32,1),grid=(1,1))
    #cuda.memcpy_dtoh(ex_get, ex_gpu)
    #cuda.memcpy_dtoh(hy_get, hy_gpu)
    #cuda.memcpy_htod(hy_gpu, hy_get)
    #cuda.memcpy_htod(ex_gpu, ex_get)
    #ex_gpu = ex_get
    #hy_gpu = hy_get
    #plt.plot(ex_get)
    #plt.ylim(-1,1)
    #plt.show()
print('GPU',time.time()-t)
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




