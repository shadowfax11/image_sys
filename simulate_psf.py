import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from components import CameraSensor, Lens, Mask
from img_sys import system_4f

IMG_SZ  = [1224, 1224]
PITCH   = [6.9e-6, 6.9e-6]
F1      = +150e-3
F2      = +50e-3
MASK_STR = 'dhpsf_optimized.mat'
MASK_SZ = [3e-3, 3e-3]
Z_START = -30e-3
Z_END   = +30e-3
NUM_Z   = 21
PSF_SZ  = [48, 48]

eps=1e-9

sensor = CameraSensor(image_size=IMG_SZ, pitch=PITCH)
lens1 = Lens(f=F1)
lens2 = Lens(f=F2)
if MASK_STR is None:
    # create a square open aperture
    H = np.zeros((1024, 1024))
    H[512-256:512+256-1,512-256:512+256+1] = 1
else:
    H = loadmat(os.path.join('./masks', MASK_STR))
    H = H['H']
mask = Mask(H, MASK_SZ)
sys_4f = system_4f(lens1, lens2, sensor, mask)

z_obj_vals = np.linspace(Z_START, Z_END, NUM_Z)
num_z = len(z_obj_vals)
psf_stack = np.zeros((sensor.img_size[0], sensor.img_size[1], num_z))

for i in range(num_z):
    z = z_obj_vals[i]
    psf_stack[:,:,i] = sys_4f.compute_PSF(z, 532e-9)

# create the figure and axes objects
fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cx = sensor.img_cntr[1]
cy = sensor.img_cntr[0]
# function that draws each frame of the animation
def animate(i):
    psf_todisp = psf_stack[cy-int(PSF_SZ[0]/2):cy+int(PSF_SZ[0]/2),cx-int(PSF_SZ[1]/2):cx+int(PSF_SZ[1]/2),i]
    ax.clear()
    cax.cla()
    im = ax.imshow(psf_todisp)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title("z = {:3.3f} mm".format(z_obj_vals[i]*1e3))
# run the animation
ani = FuncAnimation(fig, animate, frames=num_z, interval=500, repeat=False)
plt.show()