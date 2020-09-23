import numpy as np
from math import exp
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
ke = 1000
ex = np.zeros(ke)
hy = np.zeros(ke)
# Pulse parameters
kc = int(ke / 2)
t0 = 40
spread = 12
nsteps = 5000
tempos_cpu = []
for i in tqdm(range(50,1001)):    
    ex = np.zeros(i)
    hy = np.zeros(i)
    ke = i
    kc = int(i / 2)
    t = time.time()
    for time_step in range(1, i):
        for k in range(1, ke):
            ex[k] = ex[k] + 0.5 * (hy[k - 1] - hy[k])
        pulse = exp(-0.5 * ((t0 - time_step) / spread) ** 2)
        ex[kc] = pulse
        for k in range(ke - 1):
            hy[k] = hy[k] + 0.5 * (ex[k] - ex[k + 1])
    t = time.time()-t
    arq = open("tempos_cpu",'a')
    arq.writelines(str(i)+" "+str(t)+"\n")
    arq.close()
    '''
plt.ylim(-1,1)
    #plt.show()
plt.plot(ex)
plt.show()
print("CPU",time.time()-t)
'''
