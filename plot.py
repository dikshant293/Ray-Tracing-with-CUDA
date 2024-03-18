import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os,sys  

try:
    os.mkdir('result')
except:
    pass

data = []
with open(sys.argv[1],"r") as f:
    data = f.readlines()

prefix = ""
if len(sys.argv)>2:
    prefix = f"[{sys.argv[2]}]"

data=[i.strip().replace('\n','').strip().split(" ") for i in data]
G_size = int(data[0][0])
G_size = '{:,}'.format(G_size)
N_rays = int(data[0][1])
N_rays = '{:,}'.format(N_rays)
nthreads = int(data[0][2])
nblocks = int(data[0][3])
NTPB = int(data[0][4])
precision = "[SP]" if data[0][5]=='float' else '[DP]'
data.pop(0)
data = [[float(i) for i in j] for j in data]
typ = sys.argv[1].split('-')[0]
title = ''
filename = ''

if typ=='omp':
    title = f"Grid size = {precision}{G_size}x{G_size} and N_rays = {N_rays}\n[{typ.upper()}] nthreads = {nthreads}"
    filename = f"result/{prefix}{precision}{G_size}-{N_rays}-{nthreads}.png"
elif typ=='cuda':
    title = f"Grid size = {precision}{G_size}x{G_size} and N_rays = {N_rays}\n[{typ.upper()}] nblocks = {nblocks} thread_per_block = {NTPB}"
    filename = f"result/{prefix}{precision}{G_size}-{N_rays}-{typ}.png"
else:
    print("check output file, something is wrong")
    exit(-1)

plt.axis('off')
plt.title(title)
data = (data-np.min(data))/(np.max(data)-np.min(data))
plt.imshow(data, cmap='gray', interpolation='none')
plt.colorbar()
plt.savefig(filename,dpi=300)

print("Plots saved in result directory")
print(filename)