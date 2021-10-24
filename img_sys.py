import os
import sys
import numpy as np
import math
from numpy.lib.function_base import interp
from scipy.interpolate import interp2d
from components import CameraSensor, Lens, Mask
import matplotlib.pyplot as plt
import pdb

EPS = 1e-9

class system_4f:
    """
    Generic class for a 4f imaging system
    """
    def __init__(self, lens1, lens2, sensor, mask) -> None:
        """
        lens1 (Lens obj): object containing first lens properties
        lens2 (Lens obj): object containing second lens properties
        sensor (CameraSensor obj): object defining the camera sensor
        mask (Mask obj): object defining the Fourier plane mask and its characteristics
        """
        print("4f system created:")
        self.lens1 = lens1
        self.f1 = lens1.f
        print("f1: %3.3f mm" %(1000*self.f1))
        self.lens2 = lens2
        self.f2 = lens2.f
        print("f2: %3.3f mm" %(1000*self.f2))
        self.sensor = sensor
        print("Image plane sensor: %d x %d, with pixel pitch: %2.2f x %2.2f (um)" \
            %(sensor.img_size[0], sensor.img_size[1], 1e6*sensor.px_size[0], 1e6*sensor.px_size[1]))
        self.mask = mask
        print("Fourier plane mask: %3.3f x %3.3f (mm) in size" % (1e3*mask.mask_size[0], 1e3*mask.mask_size[1]))

    def compute_PSF(self, z_obj, wavelength, src_strength=1):
        """
        Computes the PSF of the given 4f system (monochromatic illumination) under the imaging condition
        z_obj (float): Distance of pt src from the first lens (in meters)
        wavelength (float): Wavelength of incident light (in meters)
        src_strength (+ve int): selects the strength (or no. of photons) emitting from the pt src
        """
        k = 2*np.pi/wavelength
        Rf2 = self.mask.X_mask**2 + self.mask.Y_mask**2
        Wdk = (k/(2*self.f1))*(1-((z_obj+self.f1)/self.f1))
        F_filt = 1 * np.exp(1j*Wdk*Rf2) * self.mask.mask
        psf = np.fft.fftshift(np.fft.fft2(F_filt))
        psf = np.abs(psf)**2
        dfx = self.mask.mask_pitch[1]
        dfy = self.mask.mask_pitch[0]
        Ix = wavelength*self.f2/dfx
        dIx = Ix/self.mask.mask.shape[1]
        Iy = wavelength*self.f2/dfy
        dIy = Iy/self.mask.mask.shape[0]
        x_img = np.linspace( -Ix/2 + dIx/2 , +Ix/2 - dIx/2 + EPS , self.mask.mask.shape[1])
        y_img = np.linspace( -Iy/2 + dIy/2 , +Iy/2 - dIy/2 + EPS , self.mask.mask.shape[0])
        Fi = interp2d(x_img, y_img, psf, kind='linear')
        psf = Fi(self.sensor.x_sensor, self.sensor.y_sensor)
        psf = src_strength*psf/np.sum(np.abs(psf))
        return psf


if __name__=="__main__":
    sensor = CameraSensor([1024, 1024], [5e-6, 5e-6])
    lens1 = Lens(150e-3)
    lens2 = Lens(50e-3)
    # create some mask (for testing)
    H = np.zeros((1024, 1024))
    H[512-256:512+256-1,512-128:512+128+1] = 1
    mask = Mask(H, [3e-3, 3e-3])
    sys_4f = system_4f(lens1, lens2, sensor, mask)
    psf = sys_4f.compute_PSF(-10e-3, 532e-9)

    plt.figure()
    plt.imshow(psf)
    plt.colorbar()
    plt.title('PSF')
    plt.show()