# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'lab_final.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_8 = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_8.sizePolicy().hasHeightForWidth())
        self.groupBox_8.setSizePolicy(sizePolicy)
        self.groupBox_8.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.groupBox_8.setTitle("")
        self.groupBox_8.setObjectName("groupBox_8")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.groupBox_8)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_8)
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.toolButton_openSource = QtWidgets.QToolButton(self.groupBox_3)
        self.toolButton_openSource.setText("")
        self.toolButton_openSource.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_openSource.setAutoRaise(False)
        self.toolButton_openSource.setObjectName("toolButton_openSource")
        self.horizontalLayout.addWidget(self.toolButton_openSource)
        self.toolButton_saveOutput = QtWidgets.QToolButton(self.groupBox_3)
        self.toolButton_saveOutput.setText("")
        self.toolButton_saveOutput.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_saveOutput.setObjectName("toolButton_saveOutput")
        self.horizontalLayout.addWidget(self.toolButton_saveOutput)
        self.toolButton_saveAsOutput = QtWidgets.QToolButton(self.groupBox_3)
        self.toolButton_saveAsOutput.setText("")
        self.toolButton_saveAsOutput.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_saveAsOutput.setObjectName("toolButton_saveAsOutput")
        self.horizontalLayout.addWidget(self.toolButton_saveAsOutput)
        self.horizontalLayout_6.addWidget(self.groupBox_3)
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_8)
        self.groupBox_5.setObjectName("groupBox_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox_5)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.toolButton_clearSource = QtWidgets.QToolButton(self.groupBox_5)
        self.toolButton_clearSource.setText("")
        self.toolButton_clearSource.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_clearSource.setAutoRaise(False)
        self.toolButton_clearSource.setObjectName("toolButton_clearSource")
        self.horizontalLayout_3.addWidget(self.toolButton_clearSource)
        self.toolButton_clearOutput = QtWidgets.QToolButton(self.groupBox_5)
        self.toolButton_clearOutput.setText("")
        self.toolButton_clearOutput.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_clearOutput.setAutoRaise(False)
        self.toolButton_clearOutput.setObjectName("toolButton_clearOutput")
        self.horizontalLayout_3.addWidget(self.toolButton_clearOutput)
        self.toolButton_undoOutput = QtWidgets.QToolButton(self.groupBox_5)
        self.toolButton_undoOutput.setText("")
        self.toolButton_undoOutput.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_undoOutput.setAutoRaise(False)
        self.toolButton_undoOutput.setObjectName("toolButton_undoOutput")
        self.horizontalLayout_3.addWidget(self.toolButton_undoOutput)
        self.toolButton_redoOutput = QtWidgets.QToolButton(self.groupBox_5)
        self.toolButton_redoOutput.setText("")
        self.toolButton_redoOutput.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_redoOutput.setAutoRaise(False)
        self.toolButton_redoOutput.setObjectName("toolButton_redoOutput")
        self.horizontalLayout_3.addWidget(self.toolButton_redoOutput)
        self.toolButton_swap = QtWidgets.QToolButton(self.groupBox_5)
        self.toolButton_swap.setText("")
        self.toolButton_swap.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_swap.setObjectName("toolButton_swap")
        self.horizontalLayout_3.addWidget(self.toolButton_swap)
        self.horizontalLayout_6.addWidget(self.groupBox_5)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_8)
        self.groupBox_4.setObjectName("groupBox_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_4)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.toolButton_rgbToGray = QtWidgets.QToolButton(self.groupBox_4)
        self.toolButton_rgbToGray.setText("")
        self.toolButton_rgbToGray.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_rgbToGray.setAutoRaise(False)
        self.toolButton_rgbToGray.setObjectName("toolButton_rgbToGray")
        self.horizontalLayout_2.addWidget(self.toolButton_rgbToGray)
        self.toolButton_rgbToHsv = QtWidgets.QToolButton(self.groupBox_4)
        self.toolButton_rgbToHsv.setText("")
        self.toolButton_rgbToHsv.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_rgbToHsv.setAutoRaise(False)
        self.toolButton_rgbToHsv.setObjectName("toolButton_rgbToHsv")
        self.horizontalLayout_2.addWidget(self.toolButton_rgbToHsv)
        self.horizontalLayout_6.addWidget(self.groupBox_4)
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox_8)
        self.groupBox_6.setObjectName("groupBox_6")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.groupBox_6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.toolButton_multiOtsu = QtWidgets.QToolButton(self.groupBox_6)
        self.toolButton_multiOtsu.setText("")
        self.toolButton_multiOtsu.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_multiOtsu.setAutoRaise(False)
        self.toolButton_multiOtsu.setObjectName("toolButton_multiOtsu")
        self.horizontalLayout_4.addWidget(self.toolButton_multiOtsu)
        self.toolButton_chanVese = QtWidgets.QToolButton(self.groupBox_6)
        self.toolButton_chanVese.setText("")
        self.toolButton_chanVese.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_chanVese.setAutoRaise(False)
        self.toolButton_chanVese.setObjectName("toolButton_chanVese")
        self.horizontalLayout_4.addWidget(self.toolButton_chanVese)
        self.toolButton_Acwe = QtWidgets.QToolButton(self.groupBox_6)
        self.toolButton_Acwe.setText("")
        self.toolButton_Acwe.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_Acwe.setAutoRaise(False)
        self.toolButton_Acwe.setObjectName("toolButton_Acwe")
        self.horizontalLayout_4.addWidget(self.toolButton_Acwe)
        self.toolButton_Gac = QtWidgets.QToolButton(self.groupBox_6)
        self.toolButton_Gac.setText("")
        self.toolButton_Gac.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_Gac.setAutoRaise(False)
        self.toolButton_Gac.setObjectName("toolButton_Gac")
        self.horizontalLayout_4.addWidget(self.toolButton_Gac)
        self.horizontalLayout_6.addWidget(self.groupBox_6)
        self.groupBox_7 = QtWidgets.QGroupBox(self.groupBox_8)
        self.groupBox_7.setObjectName("groupBox_7")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.groupBox_7)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.toolButton_roberts = QtWidgets.QToolButton(self.groupBox_7)
        self.toolButton_roberts.setText("")
        self.toolButton_roberts.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_roberts.setAutoRaise(False)
        self.toolButton_roberts.setObjectName("toolButton_roberts")
        self.horizontalLayout_5.addWidget(self.toolButton_roberts)
        self.toolButton_sobel = QtWidgets.QToolButton(self.groupBox_7)
        self.toolButton_sobel.setText("")
        self.toolButton_sobel.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_sobel.setAutoRaise(False)
        self.toolButton_sobel.setObjectName("toolButton_sobel")
        self.horizontalLayout_5.addWidget(self.toolButton_sobel)
        self.toolButton_scharr = QtWidgets.QToolButton(self.groupBox_7)
        self.toolButton_scharr.setText("")
        self.toolButton_scharr.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_scharr.setAutoRaise(False)
        self.toolButton_scharr.setObjectName("toolButton_scharr")
        self.horizontalLayout_5.addWidget(self.toolButton_scharr)
        self.toolButton_prewitt = QtWidgets.QToolButton(self.groupBox_7)
        self.toolButton_prewitt.setText("")
        self.toolButton_prewitt.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_prewitt.setAutoRaise(False)
        self.toolButton_prewitt.setObjectName("toolButton_prewitt")
        self.horizontalLayout_5.addWidget(self.toolButton_prewitt)
        self.horizontalLayout_6.addWidget(self.groupBox_7)
        self.groupBox_9 = QtWidgets.QGroupBox(self.groupBox_8)
        self.groupBox_9.setObjectName("groupBox_9")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.groupBox_9)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.toolButton_exit = QtWidgets.QToolButton(self.groupBox_9)
        self.toolButton_exit.setText("")
        self.toolButton_exit.setIconSize(QtCore.QSize(32, 32))
        self.toolButton_exit.setAutoRaise(False)
        self.toolButton_exit.setObjectName("toolButton_exit")
        self.horizontalLayout_7.addWidget(self.toolButton_exit)
        self.horizontalLayout_6.addWidget(self.groupBox_9)
        self.verticalLayout.addWidget(self.groupBox_8)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.groupBox_2 = QtWidgets.QGroupBox(self.frame)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_input = QtWidgets.QLabel(self.groupBox_2)
        self.label_input.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_input.setText("")
        self.label_input.setScaledContents(True)
        self.label_input.setObjectName("label_input")
        self.horizontalLayout_9.addWidget(self.label_input)
        self.horizontalLayout_8.addWidget(self.groupBox_2)
        self.groupBox = QtWidgets.QGroupBox(self.frame)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_output = QtWidgets.QLabel(self.groupBox)
        self.label_output.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_output.setText("")
        self.label_output.setScaledContents(True)
        self.label_output.setObjectName("label_output")
        self.horizontalLayout_10.addWidget(self.label_output)
        self.horizontalLayout_8.addWidget(self.groupBox)
        self.verticalLayout.addWidget(self.frame)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuClear = QtWidgets.QMenu(self.menuEdit)
        self.menuClear.setObjectName("menuClear")
        self.menuConversion = QtWidgets.QMenu(self.menubar)
        self.menuConversion.setObjectName("menuConversion")
        self.menuSegmentation = QtWidgets.QMenu(self.menubar)
        self.menuSegmentation.setObjectName("menuSegmentation")
        self.menuMorphological_Snakes = QtWidgets.QMenu(self.menuSegmentation)
        self.menuMorphological_Snakes.setObjectName("menuMorphological_Snakes")
        self.menuEdge_Detection_Menu = QtWidgets.QMenu(self.menubar)
        self.menuEdge_Detection_Menu.setObjectName("menuEdge_Detection_Menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setEnabled(True)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.RightToolBarArea, self.toolBar)
        self.toolBar_2 = QtWidgets.QToolBar(MainWindow)
        self.toolBar_2.setObjectName("toolBar_2")
        MainWindow.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolBar_2)
        self.actionOpenSource = QtWidgets.QAction(MainWindow)
        self.actionOpenSource.setCheckable(False)
        self.actionOpenSource.setObjectName("actionOpenSource")
        self.actionSaveOutput = QtWidgets.QAction(MainWindow)
        self.actionSaveOutput.setObjectName("actionSaveOutput")
        self.actionSaveAsOutput = QtWidgets.QAction(MainWindow)
        self.actionSaveAsOutput.setObjectName("actionSaveAsOutput")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionUndoOutput = QtWidgets.QAction(MainWindow)
        self.actionUndoOutput.setObjectName("actionUndoOutput")
        self.actionRedoOutput = QtWidgets.QAction(MainWindow)
        self.actionRedoOutput.setObjectName("actionRedoOutput")
        self.actionClearSource = QtWidgets.QAction(MainWindow)
        self.actionClearSource.setObjectName("actionClearSource")
        self.actionClearOutput = QtWidgets.QAction(MainWindow)
        self.actionClearOutput.setObjectName("actionClearOutput")
        self.actionRgbToGray = QtWidgets.QAction(MainWindow)
        self.actionRgbToGray.setObjectName("actionRgbToGray")
        self.actionRgbToHsv = QtWidgets.QAction(MainWindow)
        self.actionRgbToHsv.setObjectName("actionRgbToHsv")
        self.actionMultiOtsu = QtWidgets.QAction(MainWindow)
        self.actionMultiOtsu.setObjectName("actionMultiOtsu")
        self.actionChanVese = QtWidgets.QAction(MainWindow)
        self.actionChanVese.setObjectName("actionChanVese")
        self.actionRoberts = QtWidgets.QAction(MainWindow)
        self.actionRoberts.setObjectName("actionRoberts")
        self.actionSobel = QtWidgets.QAction(MainWindow)
        self.actionSobel.setObjectName("actionSobel")
        self.actionScharr = QtWidgets.QAction(MainWindow)
        self.actionScharr.setObjectName("actionScharr")
        self.actionPrewitt = QtWidgets.QAction(MainWindow)
        self.actionPrewitt.setObjectName("actionPrewitt")
        self.actionAcwe = QtWidgets.QAction(MainWindow)
        self.actionAcwe.setObjectName("actionAcwe")
        self.actionGac = QtWidgets.QAction(MainWindow)
        self.actionGac.setObjectName("actionGac")
        self.actionSwap = QtWidgets.QAction(MainWindow)
        self.actionSwap.setObjectName("actionSwap")
        self.menuFile.addAction(self.actionOpenSource)
        self.menuFile.addAction(self.actionSaveOutput)
        self.menuFile.addAction(self.actionSaveAsOutput)
        self.menuFile.addAction(self.actionExit)
        self.menuClear.addAction(self.actionClearSource)
        self.menuClear.addAction(self.actionClearOutput)
        self.menuEdit.addAction(self.menuClear.menuAction())
        self.menuEdit.addAction(self.actionUndoOutput)
        self.menuEdit.addAction(self.actionRedoOutput)
        self.menuEdit.addAction(self.actionSwap)
        self.menuConversion.addAction(self.actionRgbToGray)
        self.menuConversion.addAction(self.actionRgbToHsv)
        self.menuMorphological_Snakes.addAction(self.actionAcwe)
        self.menuMorphological_Snakes.addAction(self.actionGac)
        self.menuSegmentation.addAction(self.actionMultiOtsu)
        self.menuSegmentation.addAction(self.actionChanVese)
        self.menuSegmentation.addAction(self.menuMorphological_Snakes.menuAction())
        self.menuEdge_Detection_Menu.addAction(self.actionRoberts)
        self.menuEdge_Detection_Menu.addAction(self.actionSobel)
        self.menuEdge_Detection_Menu.addAction(self.actionScharr)
        self.menuEdge_Detection_Menu.addAction(self.actionPrewitt)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuConversion.menuAction())
        self.menubar.addAction(self.menuSegmentation.menuAction())
        self.menubar.addAction(self.menuEdge_Detection_Menu.menuAction())
        self.toolBar.addAction(self.actionRgbToGray)
        self.toolBar.addAction(self.actionRgbToHsv)
        self.toolBar.addAction(self.actionMultiOtsu)
        self.toolBar.addAction(self.actionChanVese)
        self.toolBar.addAction(self.actionRoberts)
        self.toolBar.addAction(self.actionSobel)
        self.toolBar.addAction(self.actionScharr)
        self.toolBar.addAction(self.actionPrewitt)
        self.toolBar.addAction(self.actionAcwe)
        self.toolBar.addAction(self.actionGac)
        self.toolBar_2.addAction(self.actionOpenSource)
        self.toolBar_2.addAction(self.actionSaveOutput)
        self.toolBar_2.addAction(self.actionSaveAsOutput)
        self.toolBar_2.addAction(self.actionClearSource)
        self.toolBar_2.addAction(self.actionClearOutput)
        self.toolBar_2.addAction(self.actionUndoOutput)
        self.toolBar_2.addAction(self.actionRedoOutput)
        self.toolBar_2.addAction(self.actionExit)

        self.retranslateUi(MainWindow)
        self.toolButton_openSource.clicked.connect(self.actionOpenSource.trigger)
        self.toolButton_saveOutput.clicked.connect(self.actionSaveOutput.trigger)
        self.toolButton_saveAsOutput.clicked.connect(self.actionSaveAsOutput.trigger)
        self.toolButton_exit.clicked.connect(self.actionExit.trigger)
        self.toolButton_chanVese.clicked.connect(self.actionChanVese.trigger)
        self.toolButton_clearSource.clicked.connect(self.actionClearSource.trigger)
        self.toolButton_clearOutput.clicked.connect(self.actionClearOutput.trigger)
        self.toolButton_Acwe.clicked.connect(self.actionAcwe.trigger)
        self.toolButton_Gac.clicked.connect(self.actionGac.trigger)
        self.toolButton_multiOtsu.clicked.connect(self.actionMultiOtsu.trigger)
        self.toolButton_prewitt.clicked.connect(self.actionPrewitt.trigger)
        self.toolButton_redoOutput.clicked.connect(self.actionRedoOutput.trigger)
        self.toolButton_rgbToGray.clicked.connect(self.actionRgbToGray.trigger)
        self.toolButton_rgbToHsv.clicked.connect(self.actionRgbToHsv.trigger)
        self.toolButton_roberts.clicked.connect(self.actionRoberts.trigger)
        self.toolButton_scharr.clicked.connect(self.actionScharr.trigger)
        self.toolButton_sobel.clicked.connect(self.actionSobel.trigger)
        self.toolButton_undoOutput.clicked.connect(self.actionUndoOutput.trigger)
        self.toolButton_swap.clicked.connect(self.actionSwap.trigger)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_3.setTitle(_translate("MainWindow", "File"))
        self.toolButton_openSource.setToolTip(_translate("MainWindow", "Open Source"))
        self.toolButton_openSource.setStatusTip(_translate("MainWindow", "Open image file (JPG or PNG)"))
        self.toolButton_saveOutput.setToolTip(_translate("MainWindow", "Save Output"))
        self.toolButton_saveOutput.setStatusTip(_translate("MainWindow", "Save output"))
        self.toolButton_saveAsOutput.setToolTip(_translate("MainWindow", "Save Output As"))
        self.toolButton_saveAsOutput.setStatusTip(_translate("MainWindow", "Save output to another location with different extension"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Edit"))
        self.toolButton_clearSource.setToolTip(_translate("MainWindow", "Clear Input Image"))
        self.toolButton_clearSource.setStatusTip(_translate("MainWindow", "Clear Input"))
        self.toolButton_clearOutput.setToolTip(_translate("MainWindow", "Clear Output Image"))
        self.toolButton_clearOutput.setStatusTip(_translate("MainWindow", "Clear Output"))
        self.toolButton_undoOutput.setToolTip(_translate("MainWindow", "Undo"))
        self.toolButton_undoOutput.setStatusTip(_translate("MainWindow", "Undo changes"))
        self.toolButton_redoOutput.setToolTip(_translate("MainWindow", "Redo"))
        self.toolButton_redoOutput.setStatusTip(_translate("MainWindow", "Redo changes"))
        self.toolButton_swap.setToolTip(_translate("MainWindow", "Swap"))
        self.toolButton_swap.setStatusTip(_translate("MainWindow", "Swap input and output images"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Conversion"))
        self.toolButton_rgbToGray.setToolTip(_translate("MainWindow", "RGB to Grayscale"))
        self.toolButton_rgbToGray.setStatusTip(_translate("MainWindow", "Convert RGB image to grayscale image"))
        self.toolButton_rgbToHsv.setToolTip(_translate("MainWindow", "RGB to Hsv"))
        self.toolButton_rgbToHsv.setStatusTip(_translate("MainWindow", "Convert RGB image to hsv image"))
        self.groupBox_6.setTitle(_translate("MainWindow", "Segmentation"))
        self.toolButton_multiOtsu.setToolTip(_translate("MainWindow", "Multi-Otsu Thresholding"))
        self.toolButton_multiOtsu.setStatusTip(_translate("MainWindow", "Apply Multi-Otsu Thresholding to input image"))
        self.toolButton_chanVese.setToolTip(_translate("MainWindow", "Chan-Vese Segmentation"))
        self.toolButton_chanVese.setStatusTip(_translate("MainWindow", "Apply Chan-Vese Segmentation to input image"))
        self.toolButton_Acwe.setToolTip(_translate("MainWindow", "Morphological Snakes ACWE"))
        self.toolButton_Acwe.setStatusTip(_translate("MainWindow", "Apply Morphological Snakes ACWE to input image"))
        self.toolButton_Gac.setToolTip(_translate("MainWindow", "Morphological Snakes GAC"))
        self.toolButton_Gac.setStatusTip(_translate("MainWindow", "Apply Morphological Snakes GAC to input image"))
        self.groupBox_7.setTitle(_translate("MainWindow", "Edge Detection"))
        self.toolButton_roberts.setToolTip(_translate("MainWindow", "Roberts"))
        self.toolButton_roberts.setStatusTip(_translate("MainWindow", "Apply Roberts edge detection"))
        self.toolButton_sobel.setToolTip(_translate("MainWindow", "Sobel"))
        self.toolButton_sobel.setStatusTip(_translate("MainWindow", "Apply Sobel edge detection"))
        self.toolButton_scharr.setToolTip(_translate("MainWindow", "Scharr"))
        self.toolButton_scharr.setStatusTip(_translate("MainWindow", "Apply Scharr edge detection"))
        self.toolButton_prewitt.setToolTip(_translate("MainWindow", "Prewitt"))
        self.toolButton_prewitt.setStatusTip(_translate("MainWindow", "Apply Prewitt edge detection"))
        self.groupBox_9.setTitle(_translate("MainWindow", "Exit"))
        self.toolButton_exit.setToolTip(_translate("MainWindow", "Exit"))
        self.toolButton_exit.setStatusTip(_translate("MainWindow", "Exit"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Input"))
        self.label_input.setStatusTip(_translate("MainWindow", "Input Image"))
        self.groupBox.setTitle(_translate("MainWindow", "Output"))
        self.label_output.setStatusTip(_translate("MainWindow", "Output Image"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuClear.setToolTip(_translate("MainWindow", "Undo"))
        self.menuClear.setStatusTip(_translate("MainWindow", "Undo changes"))
        self.menuClear.setTitle(_translate("MainWindow", "Clear"))
        self.menuConversion.setTitle(_translate("MainWindow", "Conversion"))
        self.menuSegmentation.setTitle(_translate("MainWindow", "Segmentation"))
        self.menuMorphological_Snakes.setTitle(_translate("MainWindow", "Morphological Snakes"))
        self.menuEdge_Detection_Menu.setTitle(_translate("MainWindow", "Edge Detection"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.toolBar_2.setWindowTitle(_translate("MainWindow", "toolBar_2"))
        self.actionOpenSource.setText(_translate("MainWindow", "Open Source"))
        self.actionOpenSource.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionSaveOutput.setText(_translate("MainWindow", "Save Output"))
        self.actionSaveOutput.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionSaveAsOutput.setText(_translate("MainWindow", "Save As Output"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit.setShortcut(_translate("MainWindow", "Alt+Q"))
        self.actionUndoOutput.setText(_translate("MainWindow", "Undo Output"))
        self.actionUndoOutput.setShortcut(_translate("MainWindow", "Ctrl+Shift+Z"))
        self.actionRedoOutput.setText(_translate("MainWindow", "Redo Output"))
        self.actionRedoOutput.setShortcut(_translate("MainWindow", "Ctrl+Y"))
        self.actionClearSource.setText(_translate("MainWindow", "Source"))
        self.actionClearOutput.setText(_translate("MainWindow", "Output"))
        self.actionRgbToGray.setText(_translate("MainWindow", "RGB to Grayscale"))
        self.actionRgbToHsv.setText(_translate("MainWindow", "RGB to HSV"))
        self.actionMultiOtsu.setText(_translate("MainWindow", "Multi-Otsu Thresholding"))
        self.actionChanVese.setText(_translate("MainWindow", "Chan-Vese Segmentation"))
        self.actionRoberts.setText(_translate("MainWindow", "Roberts"))
        self.actionSobel.setText(_translate("MainWindow", "Sobel"))
        self.actionScharr.setText(_translate("MainWindow", "Scharr"))
        self.actionPrewitt.setText(_translate("MainWindow", "Prewitt"))
        self.actionAcwe.setText(_translate("MainWindow", "ACWE"))
        self.actionGac.setText(_translate("MainWindow", "GAC"))
        self.actionSwap.setText(_translate("MainWindow", "Swap"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
