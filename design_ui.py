# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'design.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1440, 805)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.actionOpen = QAction(MainWindow)
        self.actionOpen.setObjectName(u"actionOpen")
        self.centralWidget = QWidget(MainWindow)
        self.centralWidget.setObjectName(u"centralWidget")
        self.layoutWidget = QWidget(self.centralWidget)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(297, 60, 812, 439))
        self.verticalLayout = QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.centralLayout = QHBoxLayout()
        self.centralLayout.setSpacing(5)
        self.centralLayout.setObjectName(u"centralLayout")
        self.graphicsView = QGraphicsView(self.layoutWidget)
        self.graphicsView.setObjectName(u"graphicsView")
        sizePolicy1 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy1)
        self.graphicsView.setMinimumSize(QSize(400, 400))
        self.graphicsView.setMaximumSize(QSize(400, 400))
        self.graphicsView.setSizeIncrement(QSize(0, 0))
        self.graphicsView.setAutoFillBackground(False)

        self.centralLayout.addWidget(self.graphicsView)

        self.graphicsView_MIP = QGraphicsView(self.layoutWidget)
        self.graphicsView_MIP.setObjectName(u"graphicsView_MIP")
        sizePolicy1.setHeightForWidth(self.graphicsView_MIP.sizePolicy().hasHeightForWidth())
        self.graphicsView_MIP.setSizePolicy(sizePolicy1)
        self.graphicsView_MIP.setMinimumSize(QSize(400, 400))
        self.graphicsView_MIP.setMaximumSize(QSize(400, 400))
        self.graphicsView_MIP.setSizeIncrement(QSize(0, 0))
        self.graphicsView_MIP.setBaseSize(QSize(400, 400))
        self.graphicsView_MIP.setAutoFillBackground(False)

        self.centralLayout.addWidget(self.graphicsView_MIP)


        self.verticalLayout.addLayout(self.centralLayout)

        self.sliderWidget = QWidget(self.layoutWidget)
        self.sliderWidget.setObjectName(u"sliderWidget")
        sizePolicy1.setHeightForWidth(self.sliderWidget.sizePolicy().hasHeightForWidth())
        self.sliderWidget.setSizePolicy(sizePolicy1)
        self.sliderWidget.setMinimumSize(QSize(810, 30))
        self.sliderWidget.setMaximumSize(QSize(810, 30))
        self.horizontalSliderZ = QSlider(self.sliderWidget)
        self.horizontalSliderZ.setObjectName(u"horizontalSliderZ")
        self.horizontalSliderZ.setGeometry(QRect(205, 0, 400, 22))
        sizePolicy1.setHeightForWidth(self.horizontalSliderZ.sizePolicy().hasHeightForWidth())
        self.horizontalSliderZ.setSizePolicy(sizePolicy1)
        self.horizontalSliderZ.setOrientation(Qt.Horizontal)

        self.verticalLayout.addWidget(self.sliderWidget)

        self.histogramWidget = QWidget(self.centralWidget)
        self.histogramWidget.setObjectName(u"histogramWidget")
        self.histogramWidget.setGeometry(QRect(29, 509, 250, 250))
        sizePolicy1.setHeightForWidth(self.histogramWidget.sizePolicy().hasHeightForWidth())
        self.histogramWidget.setSizePolicy(sizePolicy1)
        self.tableMetrics = QTableWidget(self.centralWidget)
        if (self.tableMetrics.columnCount() < 3):
            self.tableMetrics.setColumnCount(3)
        if (self.tableMetrics.rowCount() < 7):
            self.tableMetrics.setRowCount(7)
        __qtablewidgetitem = QTableWidgetItem()
        __qtablewidgetitem.setTextAlignment(Qt.AlignCenter);
        self.tableMetrics.setItem(0, 0, __qtablewidgetitem)
        self.tableMetrics.setObjectName(u"tableMetrics")
        self.tableMetrics.setGeometry(QRect(1130, 80, 241, 241))
        sizePolicy1.setHeightForWidth(self.tableMetrics.sizePolicy().hasHeightForWidth())
        self.tableMetrics.setSizePolicy(sizePolicy1)
        self.tableMetrics.setMaximumSize(QSize(256, 340))
        self.tableMetrics.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tableMetrics.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tableMetrics.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableMetrics.setTabKeyNavigation(True)
        self.tableMetrics.setProperty("showDropIndicator", True)
        self.tableMetrics.setAlternatingRowColors(False)
        self.tableMetrics.setShowGrid(False)
        self.tableMetrics.setCornerButtonEnabled(True)
        self.tableMetrics.setRowCount(7)
        self.tableMetrics.setColumnCount(3)
        self.tableMetrics.horizontalHeader().setVisible(True)
        self.tableMetrics.horizontalHeader().setCascadingSectionResizes(True)
        self.tableMetrics.horizontalHeader().setMinimumSectionSize(20)
        self.tableMetrics.horizontalHeader().setDefaultSectionSize(80)
        self.tableMetrics.horizontalHeader().setHighlightSections(True)
        self.tableMetrics.horizontalHeader().setStretchLastSection(False)
        self.tableMetrics.verticalHeader().setVisible(False)
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QMenuBar(MainWindow)
        self.menuBar.setObjectName(u"menuBar")
        self.menuBar.setGeometry(QRect(0, 0, 1440, 24))
        self.menu = QMenu(self.menuBar)
        self.menu.setObjectName(u"menu")
        MainWindow.setMenuBar(self.menuBar)

        self.menuBar.addAction(self.menu.menuAction())
        self.menu.addAction(self.actionOpen)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"oct viewer", None))
        self.actionOpen.setText(QCoreApplication.translate("MainWindow", u"Open", None))

        __sortingEnabled = self.tableMetrics.isSortingEnabled()
        self.tableMetrics.setSortingEnabled(False)
        self.tableMetrics.setSortingEnabled(__sortingEnabled)

        self.menu.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
    # retranslateUi

