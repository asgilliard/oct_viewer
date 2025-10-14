import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QMainWindow,
)

from design_ui import Ui_MainWindow


class Viewer(QMainWindow, Ui_MainWindow):
    graphicsView: QGraphicsView
    actionOpen: QAction

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Scene creation
        self.scene = QGraphicsScene(self)
        self.graphicsView.setScene(self.scene)  # graphicsView - name of QGraphicsView from .ui

        # MIP scene
        self.scene_MIP = QGraphicsScene(self)
        self.graphicsView_MIP.setScene(self.scene_MIP)

        # File -> Open menu
        self.actionOpen.triggered.connect(self.open_file)

        # Y-slider
        self.horizontalSliderY.valueChanged.connect(self.update_views)

    def update_views(self, y):
        if not hasattr(self, 'image'):
            return

        # Y-slice
        slice_y = self.image[:, y, :].T.copy()

        # Local MIP
        y_min = max(0, y - 12)
        y_max = min(512, y + 13)
        mip_y = np.max(self.image[:, y_min:y_max, :], axis=1).T.copy()

        # To QImage
        q_slice = QImage(slice_y.data, 512, 512, QImage.Format.Format_Grayscale8)
        q_mip = QImage(mip_y.data, 512, 512, QImage.Format.Format_Grayscale8)

        # To QPixmap
        pix_slice = QPixmap.fromImage(q_slice)
        pix_mip = QPixmap.fromImage(q_mip)

        # Rendering
        self.scene.clear()
        self.scene.addPixmap(pix_slice)
        self.graphicsView.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        self.scene_MIP.clear()
        self.scene_MIP.addPixmap(pix_mip)
        self.graphicsView_MIP.fitInView(
            self.scene_MIP.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Открыть .dat', '', 'DAT Files (*.dat)')
        if not file_name:
            return

        # Loading .dat like np-array 512x512x512 uint8
        data = np.fromfile(file_name, dtype=np.uint8)
        if data.size != 512 * 512 * 512:
            print('Wrong extension')
            return

        image = data.reshape((512, 512, 512)).copy()

        # Настраиваем слайдер
        self.horizontalSliderY.setMinimum(0)
        self.horizontalSliderY.setMaximum(511)
        self.horizontalSliderY.setValue(256)

        self.image = image  # saving image to attribute
        self.update_views(self.horizontalSliderY.value())


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Viewer()
    window.show()

    sys.exit(app.exec())
