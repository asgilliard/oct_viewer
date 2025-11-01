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
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.layoutWidget = QWidget(self.centralwidget)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(297, 60, 814, 484))
        self.verticalLayout = QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
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

        self.horizontalLayout.addWidget(self.graphicsView)

        self.graphicsView_MIP = QGraphicsView(self.layoutWidget)
        self.graphicsView_MIP.setObjectName(u"graphicsView_MIP")
        sizePolicy1.setHeightForWidth(self.graphicsView_MIP.sizePolicy().hasHeightForWidth())
        self.graphicsView_MIP.setSizePolicy(sizePolicy1)
        self.graphicsView_MIP.setMinimumSize(QSize(400, 400))
        self.graphicsView_MIP.setMaximumSize(QSize(400, 400))
        self.graphicsView_MIP.setSizeIncrement(QSize(0, 0))
        self.graphicsView_MIP.setBaseSize(QSize(400, 400))
        self.graphicsView_MIP.setAutoFillBackground(False)

        self.horizontalLayout.addWidget(self.graphicsView_MIP)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.widget = QWidget(self.layoutWidget)
        self.widget.setObjectName(u"widget")
        sizePolicy1.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy1)
        self.widget.setMinimumSize(QSize(810, 30))
        self.widget.setMaximumSize(QSize(810, 30))
        self.layoutWidget1 = QWidget(self.widget)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.layoutWidget1.setGeometry(QRect(10, 0, 795, 27))
        self.horizontalLayout_2 = QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer_2 = QSpacerItem(190, 17, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.horizontalSliderZ = QSlider(self.layoutWidget1)
        self.horizontalSliderZ.setObjectName(u"horizontalSliderZ")
        sizePolicy1.setHeightForWidth(self.horizontalSliderZ.sizePolicy().hasHeightForWidth())
        self.horizontalSliderZ.setSizePolicy(sizePolicy1)
        self.horizontalSliderZ.setMinimumSize(QSize(400, 25))
        self.horizontalSliderZ.setMaximumSize(QSize(400, 25))
        self.horizontalSliderZ.setOrientation(Qt.Horizontal)

        self.horizontalLayout_2.addWidget(self.horizontalSliderZ)

        self.horizontalSpacer = QSpacerItem(190, 17, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)


        self.verticalLayout.addWidget(self.widget)

        MainWindow.setCentralWidget(self.centralwidget)
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
        self.actionOpen.setText(QCoreApplication.translate("MainWindow", u"\u041e\u0442\u043a\u0440\u044b\u0442\u044c", None))
        self.menu.setTitle(QCoreApplication.translate("MainWindow", u"\u0424\u0430\u0439\u043b", None))
    # retranslateUi

