import numpy as np
from math import exp
from matplotlib import pyplot as plt
import time
ke = 1000
ex = np.zeros(ke)
hy = np.zeros(ke)
# Pulse parameters
kc = int(ke / 2)
t0 = 40
spread = 12
nsteps = 2000
tempos_cpu = []
for i in range(1000,1001):
    
    ex = np.zeros(i)
    hy = np.zeros(i)
    ke = i
    # Pulse parameters
    kc = int(i / 2)
    t = time.time()
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
            
        #plt.plot(ex)
    tempos_cpu.append(time.time()-t)
plt.ylim(-1,1)
    #plt.show()
plt.plot(ex)
plt.show()
print("CPU",time.time()-t)