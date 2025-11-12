import logging
import sys
from functools import lru_cache

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide2.QtCore import QPointF, Qt, QTimer
from PySide2.QtGui import QBrush, QColor, QImage, QPen, QPixmap
from PySide2.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QMainWindow,
    QTableWidgetItem,
    QVBoxLayout,
)

from design_ui import Ui_MainWindow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistogramCanvas(FigureCanvasQTAgg):
    """Canvas for histograms"""
    def __init__(self):
        figure = Figure(figsize=(6, 3), dpi=80)
        self.axes = figure.add_subplot(111)
        figure.tight_layout()
        super().__init__(figure)
        
    def plot_histograms(self, data1, data2):
        if data1.size == 0 or data2.size == 0:
            return
           
        self.axes.clear()
        self.axes.hist(data1, bins=50, range=(0, 256), color='red', alpha=0.5, label='ASJ')
        self.axes.hist(data2, bins=50, range=(0, 256), color='green', alpha=0.5, label='ED')
        self.axes.legend()
        self.draw()

class DraggableCircle(QGraphicsEllipseItem):
    """Draggable circle object initialisation"""
    def __init__(self, x, y, radius=30, color=None):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.setPos(x, y)
        self.radius = radius
        self.setBrush(QBrush())
        actual_color = color if color is not None else QColor(Qt.red)
        self.setPen(QPen(QBrush(), 2))
        self.setPen(QPen(actual_color))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        
        self.on_position_changed = lambda: None
        self._updating = False

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and not self._updating:
            
            scene_rect = self.scene().sceneRect()
            bounded_x = max(self.radius, min(scene_rect.width() - self.radius, value.x()))
            bounded_y = max(self.radius, min(scene_rect.height() - self.radius, value.y()))
                
            new_pos = QPointF(bounded_x, bounded_y)
            
            if value != new_pos:
                value = new_pos
                
            self.on_position_changed()
            
            return value
            
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
    """Viewer class for displaying 3D images"""
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.image = np.zeros((1, 1, 1), dtype=np.uint8)
        
        # Cache and flags
        self._pixmap_cache = {}
        self._mip_pixmap_cache = {}
        self.mask_cache = {}
        self.max_cache_size = 50
        self._current_pixmap_item = None
        self._current_mip_pixmap_item = None
        self.image_loaded = False
        
        # Scenes
        view_size = self.graphicsView.size()
        
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, view_size.width(), view_size.height())
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.scene_MIP = QGraphicsScene(self)
        self.scene_MIP.setSceneRect(0, 0, view_size.width(), view_size.height())
        self.graphicsView_MIP.setScene(self.scene_MIP)
        self.graphicsView_MIP.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.graphicsView_MIP.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Slider ranges
        self.sliderZ.setRange(0, 511)
        self.sliderMIP.setRange(2, 100)
        
        # Connections
        self.actionOpen.triggered.connect(self.open_file)
        self.sliderZ.valueChanged.connect(self.update_views)
        self.sliderMIP.valueChanged.connect(self.on_layers_changed)
        
        # Slider values
        self.sliderZ.setValue(256)
        self.sliderMIP.setValue(25)
        
        # Labels
        self.current_z_label.setText(f"Z: {self.sliderZ.value()}")
        self.MIP_layers_label.setText(f"MIP: {self.sliderMIP.value()}")

        # Circles
        self.circles = []
        self.circles_MIP = []
        
        for i, color in enumerate([QColor(255, 0, 0), QColor(0, 255, 0)]):
            idx = i
            
            # For scene
            circle = DraggableCircle(100 + i * 100, 100, color=color)
            circle.on_position_changed = lambda sender=circle, index=idx: self.sync_circle_pair(sender, index)
            self.circles.append(circle)
            self.scene.addItem(circle)
        
            # For scene_MIP
            circle_MIP = DraggableCircle(100 + i * 100, 100, color=color)
            circle_MIP.on_position_changed = lambda sender=circle_MIP, index=idx: self.sync_circle_pair(sender, index)
            self.circles_MIP.append(circle_MIP)
            self.scene_MIP.addItem(circle_MIP)
            
        # Histogram
        self.histogram_canvas = HistogramCanvas()
        histogram_layout = QVBoxLayout(self.histogramWidget)
        histogram_layout.addWidget(self.histogram_canvas)
        
        self.setup_metrics_table()
        
        # Timer
        self.histogram_update_timer = QTimer()
        self.histogram_update_timer.setSingleShot(True)
        self.histogram_update_timer.timeout.connect(self._delayed_histogram_update)

    def sync_circle_pair(self, sender, index):
        c1 = self.circles[index]
        c2 = self.circles_MIP[index]
        
        target = c2 if sender == c1 else c1
        target._updating = True
        target.setPos(sender.pos())
        target._updating = False
        
        self.update_metrics()
        self.histogram_update_timer.start(100)
    
    @lru_cache(maxsize=100)  # noqa: B019
    def compute_mip(self, z: int, layers: int = 25) -> np.ndarray:
        """Compute MIP slice with caching"""
        if self.image is None:
            return np.zeros((512, 512), dtype=np.uint8)
        
        half_layers = layers // 2
        z_min = max(0, z - half_layers)
        z_max = min(512, z + half_layers + 1)
        
        slice_data = np.array(self.image[z_min:z_max, :, :])
        return np.max(slice_data, axis=0)
    
    def get_mip_pixmap(self, z: int, layers: int = 25) -> QPixmap:
        cache_key = (z, layers)
            
        if cache_key in self._mip_pixmap_cache:
            return self._mip_pixmap_cache[cache_key]
            
        mip_data = self.compute_mip(z, layers)
            
        qimage = self.array_to_qimage(mip_data)
        pixmap = QPixmap.fromImage(qimage)
            
        if len(self._mip_pixmap_cache) > self.max_cache_size:
            self._mip_pixmap_cache.pop(next(iter(self._mip_pixmap_cache)))
                
        self._mip_pixmap_cache[cache_key] = pixmap
        return pixmap
    
    def clean_all(self):
        self._pixmap_cache.clear()
        self._mip_pixmap_cache.clear()
        self.mask_cache.clear()
        self.compute_mip.cache_clear()
        
        if hasattr(self, 'image'):
            del self.image
            self.image = None
        
        self.image_loaded = False
        
        if self._current_pixmap_item:
            self.scene.removeItem(self._current_pixmap_item)
            self._current_pixmap_item = None
            
        if self._current_mip_pixmap_item:
            self.scene.removeItem(self._current_mip_pixmap_item)
            self._current_mip_pixmap_item = None
    
    def open_file(self):
        self.clean_all()

        file_name, _ = QFileDialog.getOpenFileName(self, 'Open .dat', '', 'DAT Files (*.dat)')
        if not file_name:
            return

        try:
            self.image = np.memmap(file_name, dtype=np.uint8, mode='r', shape=(512, 512, 512))
            self._pixmap_cache.clear()  # сache reset
            self._mip_pixmap_cache.clear()  # mip cache reset
            self.image_loaded = True
            self.update_views(256)
        except Exception as e:
            logger.error(f'Failed to load file: {e}')
            self.clean_all()

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
        self.current_z_label.setText(f"Z: {z}")
        
        if not self.image_loaded or self.image is None:
            return
        
        # Cache cleanup for main slice
        keys_to_remove = [k for k in self._pixmap_cache.keys() if abs(k - z) > 10]
        for key in keys_to_remove:
            del self._pixmap_cache[key]
        
        if len(self._pixmap_cache) > self.max_cache_size:
            oldest = next(iter(self._pixmap_cache))
            del self._pixmap_cache[oldest]
        
        # Cache main slice
        if z not in self._pixmap_cache:
            slice_z = np.array(self.image[z, :, :])
            qimage = self.array_to_qimage(slice_z)
            self._pixmap_cache[z] = QPixmap.fromImage(qimage)
        
        # Rendering main slice
        if self._current_pixmap_item:
            self._current_pixmap_item.setPixmap(self._pixmap_cache[z])
        else:
            self._current_pixmap_item = self.scene.addPixmap(self._pixmap_cache[z])
            self._current_pixmap_item.setZValue(-1)
        
        self.scene.setSceneRect(0, 0, 512, 512)
        
        # MIP
        if not hasattr(self, '_current_mip_pixmap_item'):
            self._current_mip_pixmap_item = None
        
        # Getting MIP data
        layers = self.sliderMIP.value()
        mip_pixmap = self.get_mip_pixmap(z, layers)
        
        # Rendering MIP
        if self._current_mip_pixmap_item:
            self._current_mip_pixmap_item.setPixmap(mip_pixmap)
        else:
            self._current_mip_pixmap_item = self.scene_MIP.addPixmap(mip_pixmap)
            self._current_mip_pixmap_item.setZValue(-1)
        
        self.scene_MIP.setSceneRect(0, 0, 512, 512)
        
        # FitInView
        self.graphicsView.fitInView(0, 0, 512, 512, Qt.KeepAspectRatio)
        self.graphicsView_MIP.fitInView(0, 0, 512, 512, Qt.KeepAspectRatio)
        
        self.update_metrics()
        self.histogram_update_timer.start(100)

    def on_layers_changed(self, value):
        self.MIP_layers_label.setText(f"MIP: {value}")
        
        # Clear only those records where the number of layers has changed
        current_layers = value
        keys_to_remove = [k for k in self._mip_pixmap_cache.keys() if k[1] != current_layers]
        for key in keys_to_remove:
            del self._mip_pixmap_cache[key]
                
        current_z = self.sliderZ.value()
        self.update_views(current_z)        

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
        
        if not (0 <= x < 512 and 0 <= y < 512):
            return np.array([])

        if radius not in self.mask_cache:
            y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
            self.mask_cache[radius] = x_grid**2 + y_grid**2 <= radius**2
                
        mask = self.mask_cache[radius]

        x_min, x_max = max(0, x - radius), min(512, x + radius + 1)
        y_min, y_max = max(0, y - radius), min(512, y + radius + 1)
    
        region = slice_data[y_min:y_max, x_min:x_max]

        mask_cropped = mask[:region.shape[0], :region.shape[1]]
    
        return region[mask_cropped] if region.size > 0 else np.array([])
        
    def calculate_metrics(self, data):
        """Calculate metrics for a data array"""
        if data.size == 0:
            return {
                'max': 0,
                'mean': 0.0,
                'std': 0.0,
                'disp': 0.0,
                'size': 0
            }
    
        std_val = float(np.std(data))
        return {
            'max': int(np.max(data)),
            'mean': float(np.mean(data)),
            'std': std_val,
            'disp': std_val ** 2,
            'size': int(data.size),
        }

    def setup_metrics_table(self):
        self.tableMetrics.setRowCount(7)
        self.tableMetrics.setColumnCount(3)
        self.tableMetrics.setHorizontalHeaderLabels(["Metric", "ASJ", "ED"])
        
        # First column
        metrics_labels = ["X", "Y", "Max", "Mean", "Std", "Disp", "Size"]
        for i, label in enumerate(metrics_labels):
            self.tableMetrics.setItem(i, 0, QTableWidgetItem(label))
        
    def update_metrics_display(self, circle1_pos, circle2_pos, metrics1, metrics2):
        # First column (ASJ)
        self.tableMetrics.setItem(0, 1, QTableWidgetItem(str(int(circle1_pos.x()))))
        self.tableMetrics.setItem(1, 1, QTableWidgetItem(str(int(circle1_pos.y()))))
        self.tableMetrics.setItem(2, 1, QTableWidgetItem(str(metrics1['max'])))
        self.tableMetrics.setItem(3, 1, QTableWidgetItem(f"{metrics1['mean']:.1f}"))
        self.tableMetrics.setItem(4, 1, QTableWidgetItem(f"{metrics1['std']:.1f}"))
        self.tableMetrics.setItem(5, 1, QTableWidgetItem(f"{metrics1['disp']:.1f}"))
        self.tableMetrics.setItem(6, 1, QTableWidgetItem(str(metrics1['size'])))
        
        # Second column (ED)
        self.tableMetrics.setItem(0, 2, QTableWidgetItem(str(int(circle2_pos.x()))))
        self.tableMetrics.setItem(1, 2, QTableWidgetItem(str(int(circle2_pos.y()))))
        self.tableMetrics.setItem(2, 2, QTableWidgetItem(str(metrics2['max'])))
        self.tableMetrics.setItem(3, 2, QTableWidgetItem(f"{metrics2['mean']:.1f}"))
        self.tableMetrics.setItem(4, 2, QTableWidgetItem(f"{metrics2['std']:.1f}"))
        self.tableMetrics.setItem(5, 2, QTableWidgetItem(f"{metrics2['disp']:.1f}"))
        self.tableMetrics.setItem(6, 2, QTableWidgetItem(str(metrics2['size'])))
    
    def update_metrics(self):
        """Update metrics table only"""
        if not self.image_loaded:
            return
    
        current_z = self.sliderZ.value()
        layers = self.sliderMIP.value()
        mip_data = self.compute_mip(current_z, layers)
    
        circle1_data = self.get_circle_data(self.circles[0], mip_data)
        circle2_data = self.get_circle_data(self.circles[1], mip_data)
    
        metrics1 = self.calculate_metrics(circle1_data)
        metrics2 = self.calculate_metrics(circle2_data)
    
        self.update_metrics_display(
            self.circles[0].pos(), 
            self.circles[1].pos(),
            metrics1, 
            metrics2
        )
    
    def _delayed_histogram_update(self):
        self.update_histograms()
        
    def update_histograms(self):
        """Update histograms only"""
        if not self.image_loaded:
            return
        
        current_z = self.sliderZ.value()
        layers = self.sliderMIP.value()
        mip_data = self.compute_mip(current_z, layers)
        
        circle1_data = self.get_circle_data(self.circles[0], mip_data)
        circle2_data = self.get_circle_data(self.circles[1], mip_data)
        
        self.histogram_canvas.plot_histograms(circle1_data, circle2_data)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Viewer()
    window.show()
    sys.exit(app.exec_())
