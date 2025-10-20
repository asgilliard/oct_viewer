import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QGraphicsScene, QMainWindow

from design_ui import Ui_MainWindow


class Viewer(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.scene = QGraphicsScene(self)
        self.graphicsView.setScene(self.scene)
        self.scene_MIP = QGraphicsScene(self)
        self.graphicsView_MIP.setScene(self.scene_MIP)

        self.actionOpen.triggered.connect(self.open_file)
        self.horizontalSliderY.valueChanged.connect(self.update_views)

        # Кэш и флаги
        self._pixmap_cache = {}
        self._view_initialized = False

    def array_to_qimage(self, arr: np.ndarray) -> QImage:
        """Convert numpy array to QImage safely"""
        if arr.size == 0:
            return QImage()
        arr_cont = np.ascontiguousarray(arr)
        return QImage(
            arr_cont.data, arr.shape[1], arr.shape[0], arr.shape[1], QImage.Format.Format_Grayscale8
        )

    def update_views(self, y):
        if not hasattr(self, 'image'):
            return

        # Preparing data
        slice_y = self.image[:, y, :].T
        y_min, y_max = max(0, y - 12), min(512, y + 13)
        mip_y = np.max(self.image[:, y_min:y_max, :], axis=1).T

        # Caching slices
        if y not in self._pixmap_cache:
            self._pixmap_cache[y] = QPixmap.fromImage(self.array_to_qimage(slice_y))

        # Rendering
        self.scene.clear()
        self.scene.addPixmap(self._pixmap_cache[y])

        self.scene_MIP.clear()
        self.scene_MIP.addPixmap(QPixmap.fromImage(self.array_to_qimage(mip_y)))

        # First view configuration
        if not self._view_initialized:
            self.graphicsView.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.graphicsView_MIP.fitInView(
                self.scene_MIP.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
            )
            self._view_initialized = True

    def resizeEvent(self, event):
        """Update on resize event"""
        if hasattr(self, 'scene'):
            self.graphicsView.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.graphicsView_MIP.fitInView(
                self.scene_MIP.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
            )
        super().resizeEvent(event)

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open .dat', '', 'DAT Files (*.dat)')
        if not file_name:
            return

        try:
            self.image = np.memmap(file_name, dtype=np.uint8, mode='r', shape=(512, 512, 512))
            self.horizontalSliderY.setRange(0, 511)
            self.horizontalSliderY.setValue(256)
            self._pixmap_cache.clear()  # Сброс кэша
            self._view_initialized = False
            self.update_views(256)
        except Exception as e:
            print(f'Loading error: {e}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Viewer()
    window.show()
    sys.exit(app.exec())
