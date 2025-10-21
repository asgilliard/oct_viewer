import logging
import sys

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QMainWindow,
)

from design_ui import Ui_MainWindow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DraggableCircle(QGraphicsEllipseItem):
    def __init__(self, x, y, radius=30, color=QColor.red):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.setPos(x, y)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setPen(QPen(color, 2))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.radius = radius
        self.on_position_changed = lambda: None

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            scene_rect = self.scene().sceneRect()
            bounded_x = max(self.radius, min(scene_rect.width() - self.radius, value.x()))
            bounded_y = max(self.radius, min(scene_rect.height() - self.radius, value.y()))

            self.on_position_changed()

            if value.x() != bounded_x or value.y() != bounded_y:
                return QPointF(bounded_x, bounded_y)
        return super().itemChange(change, value)


class Viewer(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.scene = QGraphicsScene(self)
        view_size = self.graphicsView.size()
        self.scene.setSceneRect(0, 0, view_size.width(), view_size.height())
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.scene_MIP = QGraphicsScene(self)
        self.graphicsView_MIP.setScene(self.scene_MIP)

        self.actionOpen.triggered.connect(self.open_file)
        self.horizontalSliderY.valueChanged.connect(self.update_views)

        # Cache and flags
        self._pixmap_cache = {}
        self.image_loaded = False

        # Circles
        self.circles = []
        for i, color in enumerate([QColor(255, 0, 0), QColor(0, 255, 0)]):
            circle = DraggableCircle(100 + i * 100, 100, color=color)
            circle.on_position_changed = self.update_histograms
            self.circles.append(circle)
            self.scene.addItem(circle)

    def array_to_qimage(self, arr: np.ndarray) -> QImage:
        """Convert numpy array to QImage safely"""
        if arr.size == 0:
            return QImage()
        arr_cont = np.ascontiguousarray(arr)
        return QImage(
            arr_cont.data, arr.shape[1], arr.shape[0], arr.shape[1], QImage.Format.Format_Grayscale8
        )

    def update_views(self, y):
        if not self.image_loaded:
            return

        # Preparing data
        slice_y = self.image[:, y, :].T
        y_min, y_max = max(0, y - 12), min(512, y + 13)
        mip_y = np.max(self.image[:, y_min:y_max, :], axis=1).T

        # Caching slices
        if y not in self._pixmap_cache:
            self._pixmap_cache[y] = QPixmap.fromImage(self.array_to_qimage(slice_y))

        # Rendering
        for item in self.scene.items():
            if isinstance(item, QGraphicsPixmapItem):
                self.scene.removeItem(item)

        self.current_pixmap_item = self.scene.addPixmap(self._pixmap_cache[y])
        self.current_pixmap_item.setZValue(-1)  # to background

        self.scene.setSceneRect(0, 0, 512, 512)

        # MIP
        self.scene_MIP.clear()
        self.scene_MIP.addPixmap(QPixmap.fromImage(self.array_to_qimage(mip_y)))

        # FitInView
        self.graphicsView.fitInView(0, 0, 512, 512, Qt.AspectRatioMode.KeepAspectRatio)
        self.graphicsView_MIP.fitInView(
            self.scene_MIP.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

    def resizeEvent(self, event):
        if self.image_loaded:  # only if file loaded
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
            self._pixmap_cache.clear()  # Cache reset
            self.image_loaded = True
            self.update_views(256)
        except Exception as e:
            logger.error(f'Failed to load file: {e}')

    def update_histograms(self):
        """Update histograms when circles are moving"""
        # print(f'Circle1: {self.circle1.pos()}, Circle2: {self.circle2.pos()}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Viewer()
    window.show()
    sys.exit(app.exec())
