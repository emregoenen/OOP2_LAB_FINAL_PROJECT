from PyQt5 import QtGui, QtWidgets
from skimage import io, img_as_float, filters
from skimage.segmentation import chan_vese
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_multiotsu
import numpy as np
from functools import partial


class NotGrayError(Exception):
    pass


class NotRgbError(Exception):
    pass


class ImageOps:
    def __init__(self, ui, image, driver):
        self.input_image = image
        self.ui = ui
        self.driver = driver
        self.processed_image = None
        self.output_image = None

        # YOU CAN DELETE HERE (OR IF YOU WANT TO HANDLE RGBA IMAGES TRY TO ACHIEVE IT)
        # print(self.input_image)
        # if self.input_image.shape[2] == 4:
        #     self.input_image, sliced_part = self.input_image[:,:,:-1], self.input_image[:,:,-1] # RGBA TO RGB
        #     print(self.input_image)
        #     print(sliced_part)

        if self.first_check():
            if self.check_compatibility():
                self.process_image()
                self.change_label()
                self.change_driver_output()

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

    def check_compatibility(self): # Override this function
        pass

    def process_image(self): # Override this function
        pass

    def change_label(self):
        io.imsave("output.png", self.processed_image)
        self.output_image = io.imread("output.png")
        self.ui.label_output.setPixmap(QtGui.QPixmap("output.png"))

    def change_driver_output(self):
        self.driver.output_image = self.output_image


class Conversion(ImageOps):
    def __init__(self, ui, image, driver):
        super().__init__(ui, image, driver)

    def check_compatibility(self):
        if np.ndim(self.input_image) != 3: # Not RGB
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - NOT RGB', 'You must open a RGB image as input !')
            return False
        else:
            return True


class RgbToGray(Conversion):

    def process_image(self):
        self.processed_image = rgb2gray(self.input_image)


class RgbToHsv(Conversion):

    def process_image(self):
        self.processed_image = rgb2hsv(self.input_image)


class Segmentation(ImageOps):
    def __init__(self, ui, image, driver):
        super().__init__(ui, image, driver)

    def check_compatibility(self):
        if np.ndim(self.input_image) != 2: # Not GRAY
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - NOT GRAYSCALE', 'You must open a grayscale image as input !')
            return False
        else:
            return True


class MultiOtsu(Segmentation):

    def process_image(self):
        thresholds = threshold_multiotsu(self.input_image)
        # Using the threshold values, we generate the three regions.
        regions = np.digitize(self.input_image, bins=thresholds)
        # plot output in 'jet'
        self.processed_image = regions


class ChanVese(Segmentation):
    def process_image(self):
        image = img_as_float(self.input_image)
        # Feel free to play around with the parameters to see how they impact the result
        cv = chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
                       max_num_iter=200, dt=0.5, init_level_set="checkerboard",
                       extended_output=True)
        # input is gray and also plot output in grayscale
        self.processed_image = cv[0]


class MorphSnakes(Segmentation):
    def store_evolution_in(self, lst):
        """Returns a callback function to store the evolution of the level sets in
        the given list.
        """

        def _store(x):
            lst.append(np.copy(x))

        return _store


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


class EdgeDetection(ImageOps):
    def __init__(self, ui, image, driver):
        super().__init__(ui, image, driver)

    def check_compatibility(self):
        if np.ndim(self.input_image) != 2: # Not GRAY
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - NOT GRAYSCALE', 'You must open a grayscale image as input !')
            return False
        else:
            return True


class Roberts(EdgeDetection):
    def process_image(self):
        self.processed_image = filters.roberts(self.input_image)


class Sobel(EdgeDetection):
    def process_image(self):
        self.processed_image = filters.sobel(self.input_image)


class Scharr(EdgeDetection):
    def process_image(self):
        self.processed_image = filters.scharr(self.input_image)


class Prewitt(EdgeDetection):
    def process_image(self):
        self.processed_image = filters.prewitt(self.input_image)


class Driver:
    def __init__(self, ui):
        self.ui = ui
        self.input_image = None
        self.output_image = None
        self.save_path = None
        self.setup_icons()
        self.setup_signal_slots()

    def exit(self):
        QtWidgets.QApplication.quit()

    def setup_signal_slots(self):
        self.ui.actionOpenSource.triggered.connect(self.open_source)
        self.ui.actionSaveOutput.triggered.connect(self.save_output)
        self.ui.actionSaveAsOutput.triggered.connect(self.save_as_output)
        self.ui.actionExit.triggered.connect(self.exit)
        self.ui.actionSwap.triggered.connect(self.swap_input_output)
        self.ui.actionUndoOutput.triggered.connect(self.say_hello)
        self.ui.actionRedoOutput.triggered.connect(self.say_hello)
        self.ui.actionClearSource.triggered.connect(self.clear_source)
        self.ui.actionClearOutput.triggered.connect(self.clear_output)
        self.ui.actionRgbToGray.triggered.connect(self.rgb_to_gray)
        self.ui.actionRgbToHsv.triggered.connect(self.rgb_to_hsv)
        self.ui.actionMultiOtsu.triggered.connect(self.multi_otsu_thresholding)
        self.ui.actionChanVese.triggered.connect(self.chan_vese_segmentation)
        self.ui.actionRoberts.triggered.connect(self.roberts)
        self.ui.actionSobel.triggered.connect(self.sobel)
        self.ui.actionScharr.triggered.connect(self.scharr)
        self.ui.actionPrewitt.triggered.connect(self.prewitt)
        self.ui.actionAcwe.triggered.connect(self.morphological_snakes_ACWE)
        self.ui.actionGac.triggered.connect(self.morphological_snakes_GAC)

    def open_source(self):
        fname,_ = QtWidgets.QFileDialog.getOpenFileName(filter="Image files (*.jpg *.png)")
        self.input_image = io.imread(fname)
        self.ui.label_input.setPixmap(QtGui.QPixmap(fname))

    def swap_input_output(self):
        if (self.input_image is not None) and (self.output_image is not None):
            io.imsave("output.png", self.input_image)
            io.imsave("input.png", self.output_image)
            self.input_image, self.output_image = self.output_image, self.input_image
            self.ui.label_input.setPixmap(QtGui.QPixmap("input.png"))
            self.ui.label_output.setPixmap(QtGui.QPixmap("output.png"))
        elif (self.input_image is None) and (self.output_image is not None):
            io.imsave("input.png", self.output_image)
            self.input_image = self.output_image
            self.output_image = None
            self.ui.label_input.setPixmap(QtGui.QPixmap("input.png"))
            self.ui.label_output.clear()
        elif (self.output_image is None) and (self.input_image is not None):
            io.imsave("output.png", self.input_image)
            self.output_image = self.input_image
            self.input_image = None
            self.ui.label_output.setPixmap(QtGui.QPixmap("output.png"))
            self.ui.label_input.clear()

    def save_output(self):
        if self.output_image is not None:
            if (self.save_path is None) or (len(self.save_path) == 0):
                self.save_path,_ = QtWidgets.QFileDialog.getSaveFileName(filter="Image files (*.jpg *.png)")
                if len(self.save_path) != 0:
                    io.imsave(self.save_path, self.output_image)
            else:
                io.imsave(self.save_path, self.output_image)
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - Output is empty',
                                          'You must process input image before saving.')

    def save_as_output(self):
        if self.output_image is not None:
            self.save_path,_ = QtWidgets.QFileDialog.getSaveFileName(filter="Image files (*.jpg *.png)")
            if len(self.save_path) != 0:
                io.imsave(self.save_path, self.output_image)
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - Output is empty',
                                          'You must process input image before saving.')

    def clear_source(self):
        self.ui.label_input.clear()
        self.input_image = None

    def clear_output(self):
        self.ui.label_output.clear()
        self.output_image = None

    def rgb_to_gray(self):
        RgbToGray(self.ui, self.input_image, self)

    def rgb_to_hsv(self):
        RgbToHsv(self.ui, self.input_image, self)

    def multi_otsu_thresholding(self): # USE THIS METHOD FOR ONLY GRAYSCALE IMAGES
        MultiOtsu(self.ui, self.input_image, self)

    def chan_vese_segmentation(self):
        ChanVese(self.ui, self.input_image, self)

    def morphological_snakes_GAC(self):
        MorphGAC(self.ui, self.input_image, self)

    def morphological_snakes_ACWE(self):
        MorphACWE(self.ui, self.input_image, self)

    def roberts(self):
        Roberts(self.ui, self.input_image, self)

    def sobel(self):
        Sobel(self.ui, self.input_image, self)

    def scharr(self):
        Scharr(self.ui, self.input_image, self)

    def prewitt(self):
        Prewitt(self.ui, self.input_image, self)

    def say_hello(self):
        print("HELLO THERE !")

    def setup_icons(self):
        self.ui.toolButton_openSource.setIcon(QtGui.QIcon("resources/icons/open.png"))
        self.ui.toolButton_saveOutput.setIcon(QtGui.QIcon("resources/icons/save.png"))
        self.ui.toolButton_saveAsOutput.setIcon(QtGui.QIcon("resources/icons/saveas.png"))
        self.ui.toolButton_swap.setIcon(QtGui.QIcon("resources/icons/swap.png"))
        self.ui.toolButton_clearSource.setIcon(QtGui.QIcon("resources/icons/clear_input.png"))
        self.ui.toolButton_clearOutput.setIcon(QtGui.QIcon("resources/icons/clear_output.png"))
        self.ui.toolButton_undoOutput.setIcon(QtGui.QIcon("resources/icons/undo.png"))
        self.ui.toolButton_redoOutput.setIcon(QtGui.QIcon("resources/icons/redo.png"))
        self.ui.toolButton_rgbToGray.setIcon(QtGui.QIcon("resources/icons/grayscale.png"))
        self.ui.toolButton_rgbToHsv.setIcon(QtGui.QIcon("resources/icons/hsv.png"))
        self.ui.toolButton_multiOtsu.setIcon(QtGui.QIcon("resources/icons/multiotsu.png"))
        self.ui.toolButton_chanVese.setIcon(QtGui.QIcon("resources/icons/chanvese.png"))
        self.ui.toolButton_Acwe.setIcon(QtGui.QIcon("resources/icons/acwe.png"))
        self.ui.toolButton_Gac.setIcon(QtGui.QIcon("resources/icons/gac.png"))
        self.ui.toolButton_roberts.setIcon(QtGui.QIcon("resources/icons/roberts.png"))
        self.ui.toolButton_sobel.setIcon(QtGui.QIcon("resources/icons/sobel.png"))
        self.ui.toolButton_scharr.setIcon(QtGui.QIcon("resources/icons/scharr.png"))
        self.ui.toolButton_prewitt.setIcon(QtGui.QIcon("resources/icons/prewitt.png"))
        self.ui.toolButton_exit.setIcon(QtGui.QIcon("resources/icons/exit.png"))

        self.ui.actionOpenSource.setIcon(QtGui.QIcon("resources/icons/open.png"))
        self.ui.actionSaveOutput.setIcon(QtGui.QIcon("resources/icons/save.png"))
        self.ui.actionSaveAsOutput.setIcon(QtGui.QIcon("resources/icons/saveas.png"))
        self.ui.actionExit.setIcon(QtGui.QIcon("resources/icons/exit.png"))
        self.ui.actionSwap.setIcon(QtGui.QIcon("resources/icons/swap.png"))
        self.ui.actionUndoOutput.setIcon(QtGui.QIcon("resources/icons/undo.png"))
        self.ui.actionRedoOutput.setIcon(QtGui.QIcon("resources/icons/redo.png"))
        self.ui.actionClearSource.setIcon(QtGui.QIcon("resources/icons/input.png"))
        self.ui.actionClearOutput.setIcon(QtGui.QIcon("resources/icons/output.png"))
        self.ui.actionRgbToGray.setIcon(QtGui.QIcon("resources/icons/grayscale.png"))
        self.ui.actionRgbToHsv.setIcon(QtGui.QIcon("resources/icons/hsv.png"))
        self.ui.actionMultiOtsu.setIcon(QtGui.QIcon("resources/icons/multiotsu.png"))
        self.ui.actionChanVese.setIcon(QtGui.QIcon("resources/icons/chanvese.png"))
        self.ui.actionRoberts.setIcon(QtGui.QIcon("resources/icons/roberts.png"))
        self.ui.actionSobel.setIcon(QtGui.QIcon("resources/icons/sobel.png"))
        self.ui.actionScharr.setIcon(QtGui.QIcon("resources/icons/scharr.png"))
        self.ui.actionPrewitt.setIcon(QtGui.QIcon("resources/icons/prewitt.png"))
        self.ui.actionAcwe.setIcon(QtGui.QIcon("resources/icons/acwe.png"))
        self.ui.actionGac.setIcon(QtGui.QIcon("resources/icons/gac.png"))

        self.ui.menuClear.setIcon(QtGui.QIcon("resources/icons/clear.png"))
