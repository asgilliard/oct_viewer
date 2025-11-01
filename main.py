import logging
import sys

import numpy as np
from PySide2.QtCore import QPointF, Qt
from PySide2.QtGui import QBrush, QColor, QImage, QPen, QPixmap
from PySide2.QtWidgets import (
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
    def __init__(self, x, y, radius=30, color=None):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.setPos(x, y)
        self.setBrush(QBrush())
        actual_color = color if color is not None else QColor(Qt.red)
        self.setPen(QPen(QBrush(), 2))
        self.setPen(QPen(actual_color))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.radius = radius
        self.on_position_changed = lambda: None

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange:
            scene_rect = self.scene().sceneRect()
            bounded_x = max(self.radius, min(scene_rect.width() - self.radius, value.x()))
            bounded_y = max(self.radius, min(scene_rect.height() - self.radius, value.y()))

            self.on_position_changed()

            if value.x() != bounded_x or value.y() != bounded_y:
                return QPointF(bounded_x, bounded_y)
        return super().itemChange(change, value)

    # def wheelEvent(self, event):
    #     delta = event.delta() if hasattr(event, 'delta') else event.angleDelta().y() / 120
    #     step = 0.1
    #     new_radius = max(5, min(100, self.radius + delta * step))
    #     self.setRadius(new_radius)
    #     event.accept()

    # def setRadius(self, radius):
    #     self.radius = radius
    #     self.setRect(-radius, -radius, radius * 2, radius * 2)
    #     self.on_position_changed()


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
        self.horizontalSliderZ.valueChanged.connect(self.update_views)

        # Cache and flags
        self._pixmap_cache = {}
        self._mip_cache = {}
        self.max_cache_size = 50
        self._current_slice_data = None
        self.image_loaded = False

        # Circles
        self.circles = []
        for i, color in enumerate([QColor(255, 0, 0), QColor(0, 255, 0)]):
            circle = DraggableCircle(100 + i * 100, 100, color=color)
            circle.on_position_changed = self.update_histograms
            self.circles.append(circle)
            self.scene.addItem(circle)
    
    def compute_mip(self, z: int) -> np.ndarray:
        """Compute MIP slice with caching"""
        if z in self._mip_cache:
            return self._mip_cache[z]
        
        z_min, z_max = max(0, z - 12), min(512, z + 13)
        mip = np.max(self.image[z_min:z_max, :, :], axis=0)
        
        if len(self._mip_cache) > self.max_cache_size:
            oldest = next(iter(self._mip_cache))
            del self._mip_cache[oldest]
        
        self._mip_cache[z] = mip
        return mip
    
    def open_file(self):
        if hasattr(self, 'image'):
            del self.image

        file_name, _ = QFileDialog.getOpenFileName(self, 'Open .dat', '', 'DAT Files (*.dat)')
        if not file_name:
            return

        try:
            self.image = np.memmap(file_name, dtype=np.uint8, mode='r', shape=(512, 512, 512))
            self.horizontalSliderZ.setRange(0, 511)
            self.horizontalSliderZ.setValue(256)
            self._pixmap_cache.clear()  # Cache reset
            self.image_loaded = True
            self.update_views(256)
        except Exception as e:
            logger.error(f'Failed to load file: {e}')

    def array_to_qimage(self, arr: np.ndarray) -> QImage:
        """Convert numpy array to QImage safely"""
        if arr.size == 0:
            return QImage()

        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)

        height, width = arr.shape[0], arr.shape[1]
        bytes_per_line = width
        return QImage(arr.data, width, height, bytes_per_line, QImage.Format_Grayscale8)  # type: ignore

    def update_views(self, z):
        if not self.image_loaded:
            return
        
        # Cleanup pixmap cache
        if len(self._pixmap_cache) > self.max_cache_size:
            oldest = next(iter(self._pixmap_cache))
            del self._pixmap_cache[oldest]
        
        # Cache main slice
        if z not in self._pixmap_cache:
            slice_z = self.image[z, :, :]
            self._pixmap_cache[z] = QPixmap.fromImage(self.array_to_qimage(slice_z))
        
        # Get cached MIP as pixmap
        mip_z = self.compute_mip(z)
        mip_pixmap = QPixmap.fromImage(self.array_to_qimage(mip_z))
        
        # Rendering
        for item in self.scene.items():
            if isinstance(item, QGraphicsPixmapItem):
                self.scene.removeItem(item)
        
        self.current_pixmap_item = self.scene.addPixmap(self._pixmap_cache[z])
        self.current_pixmap_item.setZValue(-1)
        self.scene.setSceneRect(0, 0, 512, 512)
        
        # MIP
        self.scene_MIP.clear()
        self.scene_MIP.addPixmap(mip_pixmap)
        
        # FitInView
        self.graphicsView.fitInView(0, 0, 512, 512, Qt.KeepAspectRatio)
        self.graphicsView_MIP.fitInView(self.scene_MIP.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        if self.image_loaded:  # only if file loaded
            self.graphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self.graphicsView_MIP.fitInView(self.scene_MIP.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)

    def get_circle_data(self, circle, slice_data):
        """Get data inside circle"""
        pos = circle.pos()
        x, y = int(pos.x()), int(pos.y())
        radius = circle.radius

        # Creating mask for circle
        y_grid, x_grid = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        mask = x_grid**2 + y_grid**2 <= radius**2

        # Cropping region
        x_min, x_max = max(0, x - radius), min(512, x + radius + 1)
        y_min, y_max = max(0, y - radius), min(512, y + radius + 1)

        region = slice_data[y_min:y_max, x_min:x_max]
        mask_cropped = mask[: region.shape[0], : region.shape[1]]

        return region[mask_cropped] if region.size > 0 else np.array([])

    def update_metrics_display(self, metrics):
        """Update metrics display in UI"""
        # Temporary console output until we add proper widgets
        for circle_name, m in metrics.items():
            print(f'{circle_name}: max={m["max"]:.1f}, mean={m["mean"]:.1f}, std={m["std"]:.1f}')

    def update_histogram_plots(self, data1, data2):
        """Display both histograms on single chart"""
        if data1.size == 0 or data2.size == 0:
            return

        # Create histogram data
        hist1, bins = np.histogram(data1, bins=20, range=(0, 255))
        hist2, _ = np.histogram(data2, bins=20, range=(0, 255))

        # Normalize for better comparison
        hist1 = hist1 / hist1.max()
        hist2 = hist2 / hist2.max()

    def calculate_metrics(self, data):
        """Calculate metrics for a data array"""
        if data.size == 0:
            return {'max': 0, 'mean': 0, 'std': 0, 'size': 0}

        return {
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'size': int(data.size),
        }

    def update_histograms(self):
        """Update histograms when circles are moved"""
        if not self.image_loaded:
            return

        # Get current slice data
        current_y = self.horizontalSliderZ.value()
        slice_data = self.image[:, current_y, :].T  # Transpose for display

        # Extract data from both circles
        circle1_data = self.get_circle_data(self.circles[0], slice_data)
        circle2_data = self.get_circle_data(self.circles[1], slice_data)

        # Calculate metrics for both regions
        metrics = {
            'circle1': self.calculate_metrics(circle1_data),
            'circle2': self.calculate_metrics(circle2_data),
        }

        # Update UI components
        self.update_metrics_display(metrics)
        self.update_histogram_plots(circle1_data, circle2_data)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Viewer()
    window.show()
    sys.exit(app.exec_())
