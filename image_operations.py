## @package ImageOperations
#  Container file for image processing algorithms.

import numpy as np
from skimage import io, img_as_float, filters
from skimage.segmentation import chan_vese
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_multiotsu
from PyQt5 import QtGui, QtWidgets

## Base class for image processing algorithms.
class ImageOps:
    ## The constructor
    # @param image The input image to image processing
    def __init__(self, ui, image, driver):
        self.input_image = image
        self.ui = ui
        self.driver = driver
        self.processed_image = None
        self.output_image = None

        if self.first_check():
            if self.check_compatibility():
                self.process_image()
                self.change_label()
                self.change_driver_output()
                self.driver.undoable_event_happened()

    ## First check if image is compatible with image processing
    def first_check(self):

        if np.ndim(self.input_image) == 3:
            if self.input_image.shape[2] == 4:
                QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - RGBA Detected',
                                              'Cannot handle RGBA images right now. You must open a RGB or a grayscale image as input !')
                return False
            elif self.input_image.shape[2] == 3:
                return True
            else:
                QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - Unknown image type',
                                              'You must open a RGB or a grayscale image as input !')
        elif np.ndim(self.input_image) == 2:
            if len(self.input_image.shape) != 2:
                QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - Not 2d grayscale image',
                                              'This program can only convert 2D grayscale arrays !')
            else:
                return True

    ## Second check for image compatibility
    #  Override this function
    def check_compatibility(self): # Override this function
        pass

    ## Image processing is done here
    #  Override this function
    def process_image(self): # Override this function
        pass

    ## Change the output display with processed image
    def change_label(self):
        io.imsave("resources/temp/output.png", self.processed_image)
        self.output_image = io.imread("resources/temp/output.png")
        self.ui.label_output.setPixmap(QtGui.QPixmap("resources/temp/output.png"))

    ## Change the driver's output_image attribute with processed image
    def change_driver_output(self):
        self.driver.output_image = self.output_image

## Base class for Conversion operations
class Conversion(ImageOps):
    def __init__(self, ui, image, driver):
        super().__init__(ui, image, driver)

    ## Check if image is RGB
    #  If image is RGB do conversion, else don't
    def check_compatibility(self):
        if np.ndim(self.input_image) != 3: # Not RGB
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - NOT RGB', 'You must open a RGB image as input !')
            return False
        else:
            return True

## Convert RGB image to grayscale image
class RgbToGray(Conversion):

    def process_image(self):
        self.processed_image = rgb2gray(self.input_image)

## Convert RGB image to HSV
class RgbToHsv(Conversion):

    def process_image(self):
        self.processed_image = rgb2hsv(self.input_image)


## Base class for Segmentation operations
class Segmentation(ImageOps):
    def __init__(self, ui, image, driver):
        super().__init__(ui, image, driver)

    ## Check if image is grayscale
    #  If image is grayscale do conversion, else don't
    def check_compatibility(self):
        if np.ndim(self.input_image) != 2: # Not GRAY
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - NOT GRAYSCALE', 'You must open a grayscale image as input !')
            return False
        else:
            return True


## Apply MultiOtsu segmentation to image
class MultiOtsu(Segmentation):

    def process_image(self):
        thresholds = threshold_multiotsu(self.input_image)
        # Using the threshold values, we generate the three regions.
        regions = np.digitize(self.input_image, bins=thresholds)
        # plot output in 'jet'
        self.processed_image = regions


## Apply ChanVese segmentation to image
class ChanVese(Segmentation):
    def process_image(self):
        image = img_as_float(self.input_image)
        # Feel free to play around with the parameters to see how they impact the result
        cv = chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
                       max_num_iter=200, dt=0.5, init_level_set="checkerboard",
                       extended_output=True)
        # input is gray and also plot output in grayscale
        self.processed_image = cv[0]


## Base class for MorphSnakes algorithms
class MorphSnakes(Segmentation):
    def store_evolution_in(self, lst):
        """Returns a callback function to store the evolution of the level sets in
        the given list.
        """

        def _store(x):
            lst.append(np.copy(x))

        return _store


## Apply MorphSnakes(ACWE) segmentation to image
class MorphACWE(MorphSnakes):
    def process_image(self):
        image = img_as_float(self.input_image)

        # Initial level set
        init_ls = checkerboard_level_set(image.shape, 6)
        # List with intermediate results for plotting the evolution
        evolution = []
        callback = self.store_evolution_in(evolution)
        self.processed_image = morphological_chan_vese(image, num_iter=35, init_level_set=init_ls,
                                     smoothing=3, iter_callback=callback)


## Apply MorphSnakes(GAC) segmentation to image
class MorphGAC(MorphSnakes):
    def process_image(self):
        image = img_as_float(self.input_image)
        gimage = inverse_gaussian_gradient(image)

        # Initial level set
        init_ls = np.zeros(image.shape, dtype=np.int8)
        init_ls[10:-10, 10:-10] = 1
        # List with intermediate results for plotting the evolution
        evolution = []
        callback = self.store_evolution_in(evolution)
        self.processed_image = morphological_geodesic_active_contour(gimage, num_iter=230,
                                                   init_level_set=init_ls,
                                                   smoothing=1, balloon=-1,
                                                   threshold=0.69,
                                                   iter_callback=callback)


## Base class for Edge Detection algorithms
class EdgeDetection(ImageOps):
    def __init__(self, ui, image, driver):
        super().__init__(ui, image, driver)

    ## Check if image is grayscale
    #  If image is grayscale do conversion, else don't
    def check_compatibility(self):
        if np.ndim(self.input_image) != 2: # Not GRAY
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - NOT GRAYSCALE', 'You must open a grayscale image as input !')
            return False
        else:
            return True


## Apply Roberts Edge Detection to image
class Roberts(EdgeDetection):
    def process_image(self):
        self.processed_image = filters.roberts(self.input_image)


## Apply Sobel Edge Detection to image
class Sobel(EdgeDetection):
    def process_image(self):
        self.processed_image = filters.sobel(self.input_image)


## Apply Scharr Edge Detection to image
class Scharr(EdgeDetection):
    def process_image(self):
        self.processed_image = filters.scharr(self.input_image)


## Apply Prewitt Edge Detection to image
class Prewitt(EdgeDetection):
    def process_image(self):
        self.processed_image = filters.prewitt(self.input_image)