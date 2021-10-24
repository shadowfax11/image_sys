import numpy as np
from numpy.core.function_base import linspace

class CameraSensor:
    """
    Defines the camera sensor properties
    """
    def __init__(self, image_size, pitch, RGB=False, name=None, bayer_pattern=None):
        self.img_size = np.array([image_size[0], image_size[1]])    # image height, width (in pixels)
        if RGB:
            self.type = 'RGB'
            self.C = 3  # number of channels
            if bayer_pattern is not None:
                self.bayer_pattern = bayer_pattern
            else:
                self.bayer_pattern = 'RGGB'
        else:
            self.type = 'Mono'
            self.C = 1
        if len(pitch)==1:
            self.px_size = np.array([pitch, pitch])         # should be in meters
        else:
            self.px_size = np.array([pitch[0], pitch[1]])   # should be in meters
        self.name = name                                    # name of camera sensor (optional)

        # create coordinate system for image plane
        dh, dw = self.px_size[0], self.px_size[1]
        h , w = dh*self.img_size[0], dw*self.img_size[1]
        self.x_sensor = np.linspace( -w/2 + dw/2 , +w/2 - dw/2 , self.img_size[1])
        self.y_sensor = np.linspace( -h/2 + dh/2 , +h/2 - dh/2 , self.img_size[1])
        self.X_sensor, self.Y_sensor = np.meshgrid(self.x_sensor, self.y_sensor)
    
    def get_physical_sensor_size(self):
        """
        Returns the physical sensor size (in units of mm x mm)
        """
        height_mm = np.float(self.px_size[0]*self.img_size[0])*1000
        width_mm = np.float(self.px_size[1]*self.img_size[1])*1000
        return height_mm, width_mm

class Lens:
    def __init__(self, f, D=None):
        self.f = f
        self.D = D      # D is set to None if its value is irrelevant

class Mask:
    """
    Class for creating an amplitude/phase mask.
    """
    def __init__(self, mask_pattern, mask_size):
        """
        mask_pattern (numpy.ndarray): 2D array of values (real or complex) 
        mask_size (list or numpy.array): Physical size of mask (h x w). Units of meters
        mask_pattern array values should have magnitude should be between [0, 1] for realistic mask patterns. 
        """
        self.mask = mask_pattern   # mask pattern can be a complex-valued as well (numpy 2D array)
        self.mask_size = np.array([mask_size[0], mask_size[1]])
        self.mask_pitch = np.array([mask_size[0]/mask_pattern.shape[0], mask_size[1]/mask_pattern.shape[1]])
        # create coordinate system on mask-plane
        h, w = self.mask_size[0], self.mask_size[1]
        dh, dw = self.mask_pitch[0], self.mask_pitch[1]
        self.x_mask = np.linspace( -w/2 + dw/2 , +w/2 - dw/2 , num=self.mask.shape[1])
        self.y_mask = np.linspace( -h/2 + dh/2 , +h/2 - dh/2 , num=self.mask.shape[0])
        self.X_mask, self.Y_mask = np.meshgrid(self.x_mask, self.y_mask)