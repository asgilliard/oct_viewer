
################################################################################
## Form generated from reading UI file 'design.ui'
##
## Created by: Qt User Interface Compiler version 6.9.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (
    QCoreApplication,
    QMetaObject,
    QRect,
    QSize,
    Qt,
)
from PySide6.QtGui import (
    QAction,
)
from PySide6.QtWidgets import (
    QGraphicsView,
    QHBoxLayout,
    QMenu,
    QMenuBar,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)


class Ui_MainWindow:
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName('MainWindow')
        MainWindow.resize(1440, 805)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.actionOpen = QAction(MainWindow)
        self.actionOpen.setObjectName('actionOpen')
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName('centralwidget')
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName('widget')
        self.widget.setGeometry(QRect(297, 60, 814, 484))
        self.verticalLayout = QVBoxLayout(self.widget)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName('verticalLayout')
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName('horizontalLayout')
        self.graphicsView = QGraphicsView(self.widget)
        self.graphicsView.setObjectName('graphicsView')
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy1)
        self.graphicsView.setMinimumSize(QSize(400, 400))
        self.graphicsView.setMaximumSize(QSize(400, 400))
        self.graphicsView.setSizeIncrement(QSize(0, 0))
        self.graphicsView.setAutoFillBackground(False)

        self.horizontalLayout.addWidget(self.graphicsView)

        self.graphicsView_MIP = QGraphicsView(self.widget)
        self.graphicsView_MIP.setObjectName('graphicsView_MIP')
        sizePolicy1.setHeightForWidth(self.graphicsView_MIP.sizePolicy().hasHeightForWidth())
        self.graphicsView_MIP.setSizePolicy(sizePolicy1)
        self.graphicsView_MIP.setMinimumSize(QSize(400, 400))
        self.graphicsView_MIP.setMaximumSize(QSize(400, 400))
        self.graphicsView_MIP.setSizeIncrement(QSize(0, 0))
        self.graphicsView_MIP.setBaseSize(QSize(400, 400))
        self.graphicsView_MIP.setAutoFillBackground(False)

        self.horizontalLayout.addWidget(self.graphicsView_MIP)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.widget1 = QWidget(self.widget)
        self.widget1.setObjectName('widget1')
        sizePolicy1.setHeightForWidth(self.widget1.sizePolicy().hasHeightForWidth())
        self.widget1.setSizePolicy(sizePolicy1)
        self.widget1.setMinimumSize(QSize(810, 30))
        self.widget1.setMaximumSize(QSize(810, 30))
        self.widget2 = QWidget(self.widget1)
        self.widget2.setObjectName('widget2')
        self.widget2.setGeometry(QRect(10, 0, 795, 27))
        self.horizontalLayout_2 = QHBoxLayout(self.widget2)
        self.horizontalLayout_2.setObjectName('horizontalLayout_2')
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer_2 = QSpacerItem(
            190, 17, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.horizontalSliderY = QSlider(self.widget2)
        self.horizontalSliderY.setObjectName('horizontalSliderY')
        sizePolicy1.setHeightForWidth(self.horizontalSliderY.sizePolicy().hasHeightForWidth())
        self.horizontalSliderY.setSizePolicy(sizePolicy1)
        self.horizontalSliderY.setMinimumSize(QSize(400, 25))
        self.horizontalSliderY.setMaximumSize(QSize(400, 25))
        self.horizontalSliderY.setOrientation(Qt.Horizontal)

        self.horizontalLayout_2.addWidget(self.horizontalSliderY)

        self.horizontalSpacer = QSpacerItem(
            190, 17, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.verticalLayout.addWidget(self.widget1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menuBar = QMenuBar(MainWindow)
        self.menuBar.setObjectName('menuBar')
        self.menuBar.setGeometry(QRect(0, 0, 1440, 24))
        self.menu = QMenu(self.menuBar)
        self.menu.setObjectName('menu')
        MainWindow.setMenuBar(self.menuBar)

        self.menuBar.addAction(self.menu.menuAction())
        self.menu.addAction(self.actionOpen)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate('MainWindow', 'pyviewer', None))
        self.actionOpen.setText(
            QCoreApplication.translate(
                'MainWindow', '\u041e\u0442\u043a\u0440\u044b\u0442\u044c', None
            )
        )
        self.menu.setTitle(
            QCoreApplication.translate('MainWindow', '\u0424\u0430\u0439\u043b', None)
        )

    # retranslateUi
