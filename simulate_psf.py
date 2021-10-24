import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from components import CameraSensor, Lens, Mask
from img_sys import system_4f
import random
import pdb

eps=1e-9

sensor = CameraSensor(image_size=[1024, 1024], pitch=[5e-6, 5e-6])
lens1 = Lens(f=150e-3)
lens2 = Lens(f=150e-3)
H = np.zeros((1024, 1024))
H[512-256:512+256-1,512-256:512+256+1] = 1
mask = Mask(H, [3e-3, 3e-3])
sys_4f = system_4f(lens1, lens2, sensor, mask)

z_obj_vals = np.arange(-20e-3,+20e-3+eps,2e-3)
num_z = len(z_obj_vals)
psf_stack = np.zeros((sensor.img_size[0], sensor.img_size[1], num_z))

for i in range(num_z):
    z = z_obj_vals[i]
    psf_stack[:,:,i] = sys_4f.compute_PSF(z, 532e-9)

# pdb.set_trace()
# create the figure and axes objects
fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
# function that draws each frame of the animation
def animate(i):
    psf_todisp = psf_stack[:,:,i]
    ax.clear()
    cax.cla()
    im = ax.imshow(psf_todisp)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(i)
# run the animation
ani = FuncAnimation(fig, animate, frames=num_z, interval=500, repeat=False)
plt.show()