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
from dataclasses import dataclass, field
from typing import ClassVar

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
                self.driver.undoable_event_happened()

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
        io.imsave("resources/temp/output.png", self.processed_image)
        self.output_image = io.imread("resources/temp/output.png")
        self.ui.label_output.setPixmap(QtGui.QPixmap("resources/temp/output.png"))

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

@dataclass(order=True, frozen=True)
class State:
    sort_index: int = field(init=False, repr=False)
    index: ClassVar[int] = 0
    input_image: np.array = None
    output_image: np.array = None

    def __post_init__(self):
        object.__setattr__(self, 'sort_index', State.index)
        State.index = State.index + 1


class Driver:
    def __init__(self, ui, MainWindow):
        self.MainWindow = MainWindow
        self.ui = ui
        self.input_image = None
        self.output_image = None
        self.save_path = None
        self.setup_icons()
        self.setup_signal_slots()
        self.MainWindow.closeEvent = self.closeEvent
        self.undo_stack = list()
        self.redo_stack = list()
        self.undoable_event_happened()
        self.file_extension = None
        self.file_types = {"*.jpg", "*.png"}

    def open_source(self):
        try:
            fname,_ = QtWidgets.QFileDialog.getOpenFileName(filter="Image files (*.jpg *.png)")
            self.input_image = io.imread(fname)
            self.ui.label_input.setPixmap(QtGui.QPixmap(fname))
            self.undoable_event_happened()
            self.file_extension = set()
            self.file_extension.add("*." + fname.split('/')[-1].split('.')[-1])
        except ValueError:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning',
                                          "Couldn't open file.")

    def save_output(self):
        if self.output_image is not None:
            if (self.save_path is None) or (len(self.save_path) == 0):
                self.save_path,_ = QtWidgets.QFileDialog.getSaveFileName(filter="Image files (*.jpg *.png)")
                if len(self.save_path) != 0:
                    io.imsave(self.save_path, self.output_image)
                    self.check_save_buttons()
            else:
                io.imsave(self.save_path, self.output_image)
                self.check_save_buttons()
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - Output is empty',
                                          'You must process input image before saving.')

    def save_as_output(self):
        if self.output_image is not None:
            self.save_path,_ = QtWidgets.QFileDialog.getSaveFileName(filter="Image files (*.jpg *.png)")
            if len(self.save_path) != 0:
                io.imsave(self.save_path, self.output_image)
                self.check_save_buttons()
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - Output is empty',
                                          'You must process input image before saving.')

    def export_as_source(self):
        if self.input_image is not None:
            filter_str = "Image files ("
            for extension in self.file_types.difference(self.file_extension):
                filter_str += str(extension) + ' '
            filter_str = filter_str[:-1] + ')'
            self.save_path, _ = QtWidgets.QFileDialog.getSaveFileName(filter=filter_str)
            if len(self.save_path) != 0:
                io.imsave(self.save_path, self.input_image)
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - Output is empty',
                                          'You must process input image before exporting.')

    def export_as_output(self):
        if self.output_image is not None:
            filter_str = "Image files ("
            for extension in self.file_types.difference(self.file_extension):
                filter_str += str(extension) + ' '
            filter_str = filter_str[:-1] + ')'
            self.save_path, _ = QtWidgets.QFileDialog.getSaveFileName(filter=filter_str)
            if len(self.save_path) != 0:
                io.imsave(self.save_path, self.output_image)
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QDialog(), 'Warning - Output is empty',
                                          'You must process input image before exporting.')

    def undo(self):
        x = None
        try:
            if len(self.undo_stack) > 1:
                x = self.undo_stack.pop()
            else:
                print('\a')
        except IndexError:
            print('\a')
        finally:
            if x:
                self.redo_stack.append(x)
                y = self.undo_stack[-1]
                self.input_image = y.input_image
                self.output_image = y.output_image
                self.update_io_labels()
                self.check_save_buttons()
                self.check_undo_redo_buttons()
                self.check_clear_buttons()
                self.check_functionality_buttons()
                self.check_swap_button()
                self.check_export_as_buttons()

    def redo(self):
        x = None
        try:
            x = self.redo_stack.pop()
        except IndexError:
            print('\a')
        finally:
            if x:
                self.undo_stack.append(x)
                self.input_image = x.input_image
                self.output_image = x.output_image
                self.update_io_labels()
                self.check_save_buttons()
                self.check_undo_redo_buttons()
                self.check_clear_buttons()
                self.check_functionality_buttons()
                self.check_swap_button()
                self.check_export_as_buttons()

    def undoable_event_happened(self):
        self.undo_stack.append(State(self.input_image, self.output_image))
        self.redo_stack.clear()

        self.check_undo_redo_buttons()
        self.check_clear_buttons()
        self.check_save_buttons()
        self.check_functionality_buttons()
        self.check_swap_button()
        self.check_export_as_buttons()

    def update_io_labels(self):
        if self.input_image is not None:
            io.imsave("resources/temp/input.png", self.input_image)
            self.ui.label_input.setPixmap(QtGui.QPixmap("resources/temp/input.png"))
        else:
            self.ui.label_input.clear()

        if self.output_image is not None:
            io.imsave("resources/temp/output.png", self.output_image)
            self.ui.label_output.setPixmap(QtGui.QPixmap("resources/temp/output.png"))
        else:
            self.ui.label_output.clear()

    def swap_input_output(self):
        if (self.input_image is not None) and (self.output_image is not None):
            io.imsave("resources/temp/output.png", self.input_image)
            io.imsave("resources/temp/input.png", self.output_image)
            self.input_image, self.output_image = self.output_image, self.input_image
            self.ui.label_input.setPixmap(QtGui.QPixmap("resources/temp/input.png"))
            self.ui.label_output.setPixmap(QtGui.QPixmap("resources/temp/output.png"))
            self.undoable_event_happened()
        elif (self.input_image is None) and (self.output_image is not None):
            io.imsave("resources/temp/input.png", self.output_image)
            self.input_image = self.output_image
            self.output_image = None
            self.ui.label_input.setPixmap(QtGui.QPixmap("resources/temp/input.png"))
            self.ui.label_output.clear()
            self.undoable_event_happened()
        elif (self.output_image is None) and (self.input_image is not None):
            io.imsave("resources/temp/output.png", self.input_image)
            self.output_image = self.input_image
            self.input_image = None
            self.ui.label_output.setPixmap(QtGui.QPixmap("resources/temp/output.png"))
            self.ui.label_input.clear()
            self.undoable_event_happened()

    def clear_source(self):
        if self.input_image is not None:
            self.ui.label_input.clear()
            self.input_image = None
            self.undoable_event_happened()

    def clear_output(self):
        if self.output_image is not None:
            self.ui.label_output.clear()
            self.output_image = None
            self.undoable_event_happened()

    def check_export_as_buttons(self):
        if self.input_image is not None:
            self.ui.toolButton_exportAsSource.setEnabled(True)
            self.ui.actionExportAsSource.setEnabled(True)
        else:
            self.ui.toolButton_exportAsSource.setEnabled(False)
            self.ui.actionExportAsSource.setEnabled(False)

        if self.output_image is not None:
            self.ui.toolButton_exportAsOutput.setEnabled(True)
            self.ui.actionExportAsOutput.setEnabled(True)
        else:
            self.ui.toolButton_exportAsOutput.setEnabled(False)
            self.ui.actionExportAsOutput.setEnabled(False)

    def check_save_buttons(self):
        if self.output_image is not None:
            self.ui.toolButton_saveOutput.setEnabled(True)
            self.ui.toolButton_saveAsOutput.setEnabled(True)
            self.ui.actionSaveOutput.setEnabled(True)
            self.ui.actionSaveAsOutput.setEnabled(True)
        else:
            self.ui.toolButton_saveOutput.setEnabled(False)
            self.ui.toolButton_saveAsOutput.setEnabled(False)
            self.ui.actionSaveOutput.setEnabled(False)
            self.ui.actionSaveAsOutput.setEnabled(False)

    def check_clear_buttons(self):
        if self.input_image is not None:
            self.ui.toolButton_clearSource.setEnabled(True)
            self.ui.actionClearSource.setEnabled(True)
        else:
            self.ui.toolButton_clearSource.setEnabled(False)
            self.ui.actionClearSource.setEnabled(False)

        if self.output_image is not None:
            self.ui.toolButton_clearOutput.setEnabled(True)
            self.ui.actionClearOutput.setEnabled(True)
        else:
            self.ui.toolButton_clearOutput.setEnabled(False)
            self.ui.actionClearOutput.setEnabled(False)

    def check_undo_redo_buttons(self):
        if len(self.undo_stack) > 1:
            self.ui.toolButton_undoOutput.setEnabled(True)
            self.ui.actionUndoOutput.setEnabled(True)
        else:
            self.ui.toolButton_undoOutput.setEnabled(False)
            self.ui.actionUndoOutput.setEnabled(False)

        if len(self.redo_stack) > 0:
            self.ui.toolButton_redoOutput.setEnabled(True)
            self.ui.actionRedoOutput.setEnabled(True)
        else:
            self.ui.toolButton_redoOutput.setEnabled(False)
            self.ui.actionRedoOutput.setEnabled(False)

    def check_swap_button(self):
        if self.input_image is not None or self.output_image is not None:
            self.ui.toolButton_swap.setEnabled(True)
            self.ui.actionSwap.setEnabled(True)
        else:
            self.ui.toolButton_swap.setEnabled(False)
            self.ui.actionSwap.setEnabled(False)

    def check_functionality_buttons(self):
        if self.input_image is not None:
            self.ui.toolButton_rgbToGray.setEnabled(True)
            self.ui.toolButton_rgbToHsv.setEnabled(True)
            self.ui.toolButton_multiOtsu.setEnabled(True)
            self.ui.toolButton_chanVese.setEnabled(True)
            self.ui.toolButton_Acwe.setEnabled(True)
            self.ui.toolButton_Gac.setEnabled(True)
            self.ui.toolButton_roberts.setEnabled(True)
            self.ui.toolButton_sobel.setEnabled(True)
            self.ui.toolButton_scharr.setEnabled(True)
            self.ui.toolButton_prewitt.setEnabled(True)
            self.ui.actionRgbToGray.setEnabled(True)
            self.ui.actionRgbToHsv.setEnabled(True)
            self.ui.actionMultiOtsu.setEnabled(True)
            self.ui.actionChanVese.setEnabled(True)
            self.ui.actionRoberts.setEnabled(True)
            self.ui.actionSobel.setEnabled(True)
            self.ui.actionScharr.setEnabled(True)
            self.ui.actionPrewitt.setEnabled(True)
            self.ui.actionAcwe.setEnabled(True)
            self.ui.actionGac.setEnabled(True)
        else:
            self.ui.toolButton_rgbToGray.setEnabled(False)
            self.ui.toolButton_rgbToHsv.setEnabled(False)
            self.ui.toolButton_multiOtsu.setEnabled(False)
            self.ui.toolButton_chanVese.setEnabled(False)
            self.ui.toolButton_Acwe.setEnabled(False)
            self.ui.toolButton_Gac.setEnabled(False)
            self.ui.toolButton_roberts.setEnabled(False)
            self.ui.toolButton_sobel.setEnabled(False)
            self.ui.toolButton_scharr.setEnabled(False)
            self.ui.toolButton_prewitt.setEnabled(False)
            self.ui.actionRgbToGray.setEnabled(False)
            self.ui.actionRgbToHsv.setEnabled(False)
            self.ui.actionMultiOtsu.setEnabled(False)
            self.ui.actionChanVese.setEnabled(False)
            self.ui.actionRoberts.setEnabled(False)
            self.ui.actionSobel.setEnabled(False)
            self.ui.actionScharr.setEnabled(False)
            self.ui.actionPrewitt.setEnabled(False)
            self.ui.actionAcwe.setEnabled(False)
            self.ui.actionGac.setEnabled(False)

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

    def closeEvent(self, event):
        if self.ui.toolButton_saveOutput.isEnabled():
            reply = QtWidgets.QMessageBox.question(self.MainWindow, 'Unsaved Changes',
                                                   "You have unsaved changes. Do you want to save them ?", QtWidgets.QMessageBox.Yes,
                                                   QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                self.save_as_output()

        reply = QtWidgets.QMessageBox.question(self.MainWindow, 'Quit',
                                               "Are you sure to quit?", QtWidgets.QMessageBox.Yes,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def exit(self):

        if self.ui.toolButton_saveOutput.isEnabled():
            reply = QtWidgets.QMessageBox.question(self.MainWindow, 'Unsaved Changes',
                                                   "You have unsaved changes. Do you want to save them ?", QtWidgets.QMessageBox.Yes,
                                                   QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                self.save_as_output()

        reply = QtWidgets.QMessageBox.question(self.MainWindow, 'Quit',
                                               "Are you sure to quit?", QtWidgets.QMessageBox.Yes,
                                               QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            QtWidgets.QApplication.quit()

    def setup_icons(self):
        self.MainWindow.setWindowIcon(QtGui.QIcon("resources/icons/main_window_2.png"))

        self.ui.toolButton_openSource.setIcon(QtGui.QIcon("resources/icons/open.png"))
        self.ui.toolButton_saveOutput.setIcon(QtGui.QIcon("resources/icons/save.png"))
        self.ui.toolButton_saveAsOutput.setIcon(QtGui.QIcon("resources/icons/saveas.png"))
        self.ui.toolButton_exportAsSource.setIcon(QtGui.QIcon("resources/icons/export_as_source.png"))
        self.ui.toolButton_exportAsOutput.setIcon(QtGui.QIcon("resources/icons/export_as_output.png"))
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
        self.ui.actionExportAsSource.setIcon(QtGui.QIcon("resources/icons/export_as_source.png"))
        self.ui.actionExportAsOutput.setIcon(QtGui.QIcon("resources/icons/export_as_output.png"))
        self.ui.actionExit.setIcon(QtGui.QIcon("resources/icons/exit.png"))
        self.ui.actionSwap.setIcon(QtGui.QIcon("resources/icons/swap.png"))
        self.ui.actionUndoOutput.setIcon(QtGui.QIcon("resources/icons/undo.png"))
        self.ui.actionRedoOutput.setIcon(QtGui.QIcon("resources/icons/redo.png"))
        self.ui.actionClearSource.setIcon(QtGui.QIcon("resources/icons/clear_input.png"))
        self.ui.actionClearOutput.setIcon(QtGui.QIcon("resources/icons/clear_output.png"))
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

    def setup_signal_slots(self):
        self.ui.actionOpenSource.triggered.connect(self.open_source)
        self.ui.actionSaveOutput.triggered.connect(self.save_output)
        self.ui.actionSaveAsOutput.triggered.connect(self.save_as_output)
        self.ui.actionExportAsSource.triggered.connect(self.export_as_source)
        self.ui.actionExportAsOutput.triggered.connect(self.export_as_output)
        self.ui.actionExit.triggered.connect(self.exit)
        self.ui.actionSwap.triggered.connect(self.swap_input_output)
        self.ui.actionUndoOutput.triggered.connect(self.undo)
        self.ui.actionRedoOutput.triggered.connect(self.redo)
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