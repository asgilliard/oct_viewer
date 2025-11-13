"""
Refactored medical image viewer with proper architecture.
"""
import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

import darkdetect
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


# Constants
IMAGE_SIZE = 512
DEFAULT_Z = 256
DEFAULT_MIP_LAYERS = 25
DEFAULT_CIRCLE_RADIUS = 30
MAX_PIXMAP_CACHE_SIZE = 50
HISTOGRAM_UPDATE_DELAY_MS = 100


class CircleType(Enum):
    """Types of measurement circles."""
    
    ASJ = 0
    ED = 1


@dataclass
class Metrics:
    """Metrics calculated from image data."""
    
    max_value: int = 0
    mean: float = 0.0
    std: float = 0.0
    variance: float = 0.0
    size: int = 0

    @classmethod
    def from_data(cls, data: np.ndarray) -> 'Metrics':
        """Fabric: calculates metrics from data array."""
        if data.size == 0:
            return cls()
        
        std_val = float(np.std(data))
        return cls(
            max_value=int(np.max(data)),
            mean=float(np.mean(data)),
            std=std_val,
            variance=std_val ** 2,
            size=int(data.size)
        )


class ImageDataManager:
    """Manages 3D image data loading and access."""
    
    def __init__(self):
        self.data: Optional[np.ndarray] = None
        self._mip_cache: Dict[Tuple[int, int], np.ndarray] = {}
        
    def load(self, file_path: Path) -> bool:
        """Load image data from file."""
        try:
            self.data = np.memmap(
                str(file_path),
                dtype=np.uint8,
                mode='r',
                shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)
            )
            self._mip_cache.clear()
            logger.info(f"Loaded image: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            return False
    
    def get_slice(self, z: int) -> Optional[np.ndarray]:
        """Get 2D slice at z position."""
        if self.data is None or not self._is_valid_z(z):
            return None
        return np.array(self.data[z, :, :])
    
    def compute_mip(self, z: int, layers: int) -> Optional[np.ndarray]:
        """Compute Maximum Intensity Projection with instance-level caching."""
        if self.data is None or not self._is_valid_z(z):
            return None
    
        # key for cache for z and layers
        key = (z, layers)
    
        if key not in self._mip_cache:
            half_layers = layers // 2
            z_min = max(0, z - half_layers)
            z_max = min(IMAGE_SIZE, z + half_layers + 1)
            
            slice_data = np.array(self.data[z_min:z_max, :, :])
            self._mip_cache[key] = np.max(slice_data, axis=0)
    
        return self._mip_cache[key]
    
    def _is_valid_z(self, z: int) -> bool:
        """Check if z coordinate is valid."""
        return 0 <= z < IMAGE_SIZE
    
    def clear(self):
        """Clear all data and caches."""
        if self.data is not None:
            del self.data
            self.data = None
        self._mip_cache.clear()


class PixmapCache:
    """Cache for QPixmap objects with LRU eviction."""
    
    def __init__(self, max_size: int = MAX_PIXMAP_CACHE_SIZE):
        self._cache: OrderedDict = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        """Get pixmap from cache."""
        pixmap = self._cache.pop(key, None)
        if pixmap is not None:
            self._cache[key] = pixmap  # move to end (mark as recently used)
        return pixmap
    
    def put(self, key, pixmap):
        if key in self._cache:
            self._cache.pop(key)
        elif len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = pixmap
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()
    
    def cleanup_around(self, key: int, keep_range: int = 10):
        """Remove keys far from current key."""
        if not isinstance(key, int):
            return
        keys_to_remove = [
            k for k in self._cache.keys() 
            if isinstance(k, int) and abs(k - key) > keep_range
        ]
        for k in keys_to_remove:
            del self._cache[k]
    
    def remove_by_condition(self, condition):
        """Remove items matching condition."""
        keys_to_remove = [k for k in self._cache.keys() if condition(k)]
        for k in keys_to_remove:
            del self._cache[k]


class Analytics:
    """Analytical calculations on image data."""
    
    def __init__(self):
        self._mask_cache: Dict[int, np.ndarray] = {}
    
    def get_circle_data(
        self, 
        x: int, 
        y: int, 
        radius: int, 
        slice_data: np.ndarray
    ) -> np.ndarray:
        """Extract data inside circle region."""
        if not (0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE):
            return np.array([])
        
        if radius not in self._mask_cache:
            self._mask_cache[radius] = self._create_circle_mask(radius)
        
        mask = self._mask_cache[radius]
        
        x_min = max(0, x - radius)
        x_max = min(IMAGE_SIZE, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(IMAGE_SIZE, y + radius + 1)
        
        region = slice_data[y_min:y_max, x_min:x_max]
        
        if region.size == 0:
            return np.array([])
        
        mask_cropped = mask[:region.shape[0], :region.shape[1]]
        
        return region[mask_cropped]
    
    @staticmethod
    def _create_circle_mask(radius: int) -> np.ndarray:
        """Create circular mask."""
        y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
        return x_grid**2 + y_grid**2 <= radius**2
    
    def clear_cache(self):
        """Clear mask cache."""
        self._mask_cache.clear()


# class HistogramCanvas(FigureCanvasQTAgg):
#     """Canvas for displaying histograms."""
    
#     def __init__(self):
#         figure = Figure(figsize=(3, 3), dpi=80)
#         self.axes = figure.add_subplot(111)
#         figure.tight_layout()
#         super().__init__(figure)
    
#     @staticmethod
#     def get_system_theme():
#         """Detect system theme (light/dark)."""
#         return darkdetect.theme()  # 'Dark' or 'Light'
    
#     def setup_plot_style(self):
#         is_dark = get_system_theme() == 'Dark'
        
#         if is_dark:
#             self.figure.style.use('dark_background')
    
#     def plot_histograms(self, data1: np.ndarray, data2: np.ndarray):
#         """Plot two histograms."""
#         if data1.size == 0 or data2.size == 0:
#             return
        
#         self.axes.clear()
#         self.axes.hist(
#             data1, bins=256, range=(0, 256), 
#             color='red', alpha=0.5, label='ASJ', edgecolor='black' 
#         )
#         self.axes.hist(
#             data2, bins=256, range=(0, 256), 
#             color='green', alpha=0.5, label='ED'
#         )
#         self.axes.legend()
#         self.draw()


class HistogramCanvas(FigureCanvasQTAgg):
    """Canvas for displaying histograms."""
    
    def __init__(self):
        figure = Figure(figsize=(3, 3), dpi=80)
        self.axes = figure.add_subplot(111)
        figure.tight_layout()
        super().__init__(figure)
        self._setup_style()
    
    @staticmethod
    def get_system_theme():
        """Detect system theme (light/dark)."""
        try:
            return darkdetect.theme()
        except Exception:
            return 'Light'
    
    def _setup_style(self):
        """Apply theme-based styling."""
        is_dark = self.get_system_theme() == 'Dark'
        
        if is_dark:
            bg_color = '#2b2b2b'
            fg_color = '#ffffff'
            grid_color = '#404040'
        else:
            bg_color = '#ffffff'
            fg_color = '#000000'
            grid_color = '#e0e0e0'
        
        self.figure.set_facecolor(bg_color)
        self.axes.set_facecolor(bg_color)
        self.axes.spines['bottom'].set_color(fg_color)
        self.axes.spines['top'].set_color(fg_color)
        self.axes.spines['left'].set_color(fg_color)
        self.axes.spines['right'].set_color(fg_color)
        self.axes.tick_params(colors=fg_color)
        self.axes.xaxis.label.set_color(fg_color)
        self.axes.yaxis.label.set_color(fg_color)
        self.axes.grid(True, alpha=0.2, color=grid_color)
    
    def plot_histograms(self, data1: np.ndarray, data2: np.ndarray):
        """Plot two histograms."""
        if data1.size == 0 or data2.size == 0:
            return
        
        self.axes.clear()
        self._setup_style()
        
        self.axes.hist(
            data1, bins=256, range=(0, 256), 
            color='#ff4444', alpha=0.6, label='ASJ'
        )
        self.axes.hist(
            data2, bins=256, range=(0, 256), 
            color='#44ff44', alpha=0.6, label='ED'
        )
        
        is_dark = self.get_system_theme() == 'Dark'
        legend = self.axes.legend(facecolor='#2b2b2b' if is_dark else 'white',
                                   edgecolor='white' if is_dark else 'black')
        for text in legend.get_texts():
            text.set_color('white' if is_dark else 'black')
        
        self.draw()


class DraggableCircle(QGraphicsEllipseItem):
    """Draggable circle with position change signals."""
    
    def __init__(
        self, 
        x: float, 
        y: float, 
        radius: int = DEFAULT_CIRCLE_RADIUS, 
        color: Optional[QColor] = None
    ):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.setPos(x, y)
        self.radius = radius
        
        self.setBrush(QBrush())
        actual_color = color if color is not None else QColor(Qt.red)
        self.setPen(QPen(actual_color, 2))
        
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        
        self._is_syncing = False
        self._position_callback = None
    
    def set_position_callback(self, callback):
        """Set callback for position changes."""
        self._position_callback = callback
    
    def itemChange(self, change, value):
        """Handle item changes."""
        if change == QGraphicsItem.ItemPositionChange and not self._is_syncing:
            scene_rect = self.scene().sceneRect()
            
            # Constrain to scene bounds
            bounded_x = max(
                self.radius, 
                min(scene_rect.width() - self.radius, value.x())
            )
            bounded_y = max(
                self.radius, 
                min(scene_rect.height() - self.radius, value.y())
            )
            
            new_pos = QPointF(bounded_x, bounded_y)
            
            if self._position_callback:
                self._position_callback()
            
            return new_pos if value != new_pos else value
        
        return super().itemChange(change, value)
    
    def sync_position(self, pos: QPointF):
        """Sync position without triggering callbacks."""
        self._is_syncing = True
        self.setPos(pos)
        self._is_syncing = False


class CircleManager:
    """Manages pairs of synchronized circles."""
    
    def __init__(self):
        self.circles: Dict[CircleType, DraggableCircle] = {}
        self.circles_mip: Dict[CircleType, DraggableCircle] = {}
        self._sync_callback = None
    
    def create_circles(
        self, 
        scene: QGraphicsScene, 
        scene_mip: QGraphicsScene
    ):
        """Create circle pairs for both scenes."""
        colors = {
            CircleType.ASJ: QColor(255, 0, 0),
            CircleType.ED: QColor(0, 255, 0)
        }
        
        for i, circle_type in enumerate(CircleType):
            x_pos = 100 + i * 100
            
            # Main scene circle
            circle = DraggableCircle(x_pos, 100, color=colors[circle_type])
            circle.set_position_callback(
                lambda ct=circle_type: self._on_position_changed(ct, False)
            )
            self.circles[circle_type] = circle
            scene.addItem(circle)
            
            # MIP scene circle
            circle_mip = DraggableCircle(x_pos, 100, color=colors[circle_type])
            circle_mip.set_position_callback(
                lambda ct=circle_type: self._on_position_changed(ct, True)
            )
            self.circles_mip[circle_type] = circle_mip
            scene_mip.addItem(circle_mip)
    
    def set_sync_callback(self, callback):
        """Set callback for position synchronization."""
        self._sync_callback = callback
    
    def _on_position_changed(self, circle_type: CircleType, is_mip: bool):
        """Handle position change of a circle."""
        source = self.circles_mip[circle_type] if is_mip else self.circles[circle_type]
        target = self.circles[circle_type] if is_mip else self.circles_mip[circle_type]
        
        target.sync_position(source.pos())
        
        if self._sync_callback:
            self._sync_callback()
    
    def get_position(self, circle_type: CircleType) -> Tuple[int, int]:
        """Get circle position."""
        pos = self.circles[circle_type].pos()
        return int(pos.x()), int(pos.y())
    
    def get_radius(self, circle_type: CircleType) -> int:
        """Get circle radius."""
        return self.circles[circle_type].radius


class MetricsTable:
    """Manages metrics display table."""
    
    METRIC_LABELS = ["X", "Y", "Max", "Mean", "Std", "Var", "Size"]
    
    def __init__(self, table_widget):
        self.table = table_widget
        self._setup_table()
    
    def _setup_table(self):
        """Initialize table structure."""
        self.table.setRowCount(len(self.METRIC_LABELS))
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Metric", "ASJ", "ED"])
        
        for i, label in enumerate(self.METRIC_LABELS):
            self.table.setItem(i, 0, QTableWidgetItem(label))
    
    def update(
        self, 
        pos_asj: Tuple[int, int], 
        pos_ed: Tuple[int, int],
        metrics_asj: Metrics, 
        metrics_ed: Metrics
    ):
        """Update table with new metrics."""
        self._update_column(1, pos_asj, metrics_asj)
        self._update_column(2, pos_ed, metrics_ed)
    
    def _update_column(self, col: int, pos: Tuple[int, int], metrics: Metrics):
        """Update single column."""
        self.table.setItem(0, col, QTableWidgetItem(str(pos[0])))
        self.table.setItem(1, col, QTableWidgetItem(str(pos[1])))
        self.table.setItem(2, col, QTableWidgetItem(str(metrics.max_value)))
        self.table.setItem(3, col, QTableWidgetItem(f"{metrics.mean:.1f}"))
        self.table.setItem(4, col, QTableWidgetItem(f"{metrics.std:.1f}"))
        self.table.setItem(5, col, QTableWidgetItem(f"{metrics.variance:.1f}"))
        self.table.setItem(6, col, QTableWidgetItem(str(metrics.size)))


# class ImageRenderer:
#     """Handles image rendering to Qt scenes."""
    
#     def __init__(self):
#         # Cache for QPixmap objects (UI layer)
#         self._pixmap_cache = PixmapCache()
#         self._mip_pixmap_cache = PixmapCache()
    
#     @staticmethod
#     def array_to_qimage(arr: np.ndarray) -> QImage:
#         """Convert numpy array to QImage."""
#         if arr.size == 0:
#             return QImage()
        
#         if not arr.flags['C_CONTIGUOUS']:
#             arr = np.ascontiguousarray(arr)
        
#         height, width = arr.shape[0], arr.shape[1]
#         bytes_per_line = width
#         return QImage(
#             arr.data, width, height,  # type: ignore
#             bytes_per_line, QImage.Format_Grayscale8
#         )  
    
#     def get_slice_pixmap(self, slice_data: np.ndarray, z: int) -> QPixmap:
#         """Get or create pixmap for slice."""
#         cached = self._pixmap_cache.get(z)
#         if cached:
#             return cached
        
#         qimage = self.array_to_qimage(slice_data)
#         pixmap = QPixmap.fromImage(qimage)
#         self._pixmap_cache.put(z, pixmap)
#         return pixmap
    
#     def get_mip_pixmap(self, mip_data: np.ndarray, z: int, layers: int) -> QPixmap:
#         """Get or create pixmap for MIP."""
#         cache_key = (z, layers)
#         cached = self._mip_pixmap_cache.get(cache_key)
#         if cached:
#             return cached
        
#         qimage = self.array_to_qimage(mip_data)
#         pixmap = QPixmap.fromImage(qimage)
#         self._mip_pixmap_cache.put(cache_key, pixmap)
#         return pixmap
    
#     def cleanup_caches(self, current_z: int, current_layers: int):
#         """Clean up old cache entries."""
#         self._pixmap_cache.cleanup_around(current_z)
        
#         self._mip_pixmap_cache.remove_by_condition(
#             lambda k: isinstance(k, tuple) and k[1] != current_layers
#         )
    
#     def clear(self):
#         """Clear all caches."""
#         self._pixmap_cache.clear()
#         self._mip_pixmap_cache.clear()

class ImageRenderer:
    """Handles image rendering to Qt scenes."""
    
    def __init__(self):
        self._pixmap_cache = PixmapCache()
        self._mip_pixmap_cache = PixmapCache()
    
    @staticmethod
    def array_to_qimage(arr: np.ndarray, invert: bool = False) -> QImage:
        """Convert numpy array to QImage."""
        if arr.size == 0:
            return QImage()
        
        if invert:
            arr = 255 - arr
        
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        
        height, width = arr.shape[0], arr.shape[1]
        bytes_per_line = width
        return QImage(
            arr.data, width, height,  # type: ignore
            bytes_per_line, QImage.Format_Grayscale8
        )  
    
    def get_slice_pixmap(self, slice_data: np.ndarray, z: int) -> QPixmap:
        """Get or create pixmap for slice."""
        invert = HistogramCanvas.get_system_theme() == 'Light'
        cache_key = (z, invert)
        
        cached = self._pixmap_cache.get(cache_key)
        if cached:
            return cached
        
        qimage = self.array_to_qimage(slice_data, invert=invert)
        pixmap = QPixmap.fromImage(qimage)
        self._pixmap_cache.put(cache_key, pixmap)
        return pixmap
    
    def get_mip_pixmap(self, mip_data: np.ndarray, z: int, layers: int) -> QPixmap:
        """Get or create pixmap for MIP."""
        invert = HistogramCanvas.get_system_theme() == 'Light'
        cache_key = (z, layers, invert)
        
        cached = self._mip_pixmap_cache.get(cache_key)
        if cached:
            return cached
        
        qimage = self.array_to_qimage(mip_data, invert=invert)
        pixmap = QPixmap.fromImage(qimage)
        self._mip_pixmap_cache.put(cache_key, pixmap)
        return pixmap
        
    def cleanup_caches(self, current_z: int, current_layers: int):
         """Clean up old cache entries."""
         self._pixmap_cache.cleanup_around(current_z)
            
         self._mip_pixmap_cache.remove_by_condition(
            lambda k: isinstance(k, tuple) and k[1] != current_layers
        )
        
    def clear(self):
        """Clear all caches."""
        self._pixmap_cache.clear()
        self._mip_pixmap_cache.clear()

class Viewer(QMainWindow, Ui_MainWindow):
    """Main viewer window."""
    
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # Components
        self.data_manager = ImageDataManager()
        self.analytics = Analytics()
        self.renderer = ImageRenderer()
        self.circle_manager = CircleManager()
        
        # UI components
        self._current_slice_item = None
        self._current_mip_item = None
        
        # Setup
        self._setup_scenes()
        self._setup_sliders()
        self._setup_histogram()
        self._setup_metrics_table()
        self._setup_circles()
        self._setup_connections()
        self._setup_timer()
        
        # Theme change handling
        self._current_theme = HistogramCanvas.get_system_theme()
        self._theme_timer = QTimer()
        self._theme_timer.timeout.connect(self._check_theme)
        self._theme_timer.start(1000)  # check every second
            
    # def _check_theme(self):
    #     """Check for theme changes."""
    #     new_theme = HistogramCanvas.get_system_theme()
    #     if new_theme != self._current_theme:
    #         self._current_theme = new_theme
    #         self.histogram_canvas._setup_style()
    #         self.histogram_canvas.draw()
    
    def _check_theme(self):
        """Check for theme changes."""
        new_theme = HistogramCanvas.get_system_theme()
        if new_theme != self._current_theme:
            self._current_theme = new_theme
            self.renderer.clear()  # очистить кэш пиксмапов
            self.update_views(self.sliderZ.value())  # перерисовать
    
    def _setup_scenes(self):
        """Initialize graphics scenes."""
        view_size = self.graphicsView.size()
        
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, view_size.width(), view_size.height())
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.scene_mip = QGraphicsScene(self)
        self.scene_mip.setSceneRect(0, 0, view_size.width(), view_size.height())
        self.graphicsView_MIP.setScene(self.scene_mip)
        self.graphicsView_MIP.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_MIP.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    
    def _setup_sliders(self):
        """Initialize sliders."""
        self.sliderZ.setRange(0, IMAGE_SIZE - 1)
        self.sliderZ.setValue(DEFAULT_Z)
        self.sliderMIP.setRange(2, 100)
        self.sliderMIP.setValue(DEFAULT_MIP_LAYERS)
        
        self.current_z_label.setText(f"Z: {DEFAULT_Z}")
        self.MIP_layers_label.setText(f"MIP: {DEFAULT_MIP_LAYERS}")
    
    def _setup_histogram(self):
        """Initialize histogram canvas."""
        self.histogram_canvas = HistogramCanvas()
        histogram_layout = QVBoxLayout(self.histogramWidget)
        histogram_layout.addWidget(self.histogram_canvas)
    
    def _setup_metrics_table(self):
        """Initialize metrics table."""
        self.metrics_table = MetricsTable(self.tableMetrics)
    
    def _setup_circles(self):
        """Initialize measurement circles."""
        self.circle_manager.create_circles(self.scene, self.scene_mip)
        self.circle_manager.set_sync_callback(self._on_circles_changed)
    
    def _setup_connections(self):
        """Connect signals and slots."""
        self.actionOpen.triggered.connect(self.open_file)
        self.sliderZ.valueChanged.connect(self.on_z_changed)
        self.sliderMIP.valueChanged.connect(self.on_layers_changed)
    
    def _setup_timer(self):
        """Initialize update timer."""
        self.histogram_timer = QTimer()
        self.histogram_timer.setSingleShot(True)
        self.histogram_timer.timeout.connect(self._update_histogram)
    
    def open_file(self):
        """Open and load image file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Open .dat', '', 'DAT Files (*.dat)'
        )
        if not file_name:
            return
        
        self._clear_all()
        
        if self.data_manager.load(Path(file_name)):
            self.update_views(DEFAULT_Z)
    
    def _clear_all(self):
        """Clear all data and caches."""
        self.data_manager.clear()
        self.analytics.clear_cache()
        self.renderer.clear()
        
        if self._current_slice_item:
            self.scene.removeItem(self._current_slice_item)
            self._current_slice_item = None
        
        if self._current_mip_item:
            self.scene_mip.removeItem(self._current_mip_item)
            self._current_mip_item = None
    
    def on_z_changed(self, z: int):
        """Handle Z slider change."""
        self.current_z_label.setText(f"Z: {z}")
        self.update_views(z)
    
    def on_layers_changed(self, layers: int):
        """Handle MIP layers slider change."""
        self.MIP_layers_label.setText(f"MIP: {layers}")
        self.renderer.cleanup_caches(self.sliderZ.value(), layers)
        self.update_views(self.sliderZ.value())
    
    def update_views(self, z: int):
        """Update all views for given Z position."""
        
        # Render main slice
        slice_data = self.data_manager.get_slice(z)
        if slice_data is not None:
            pixmap = self.renderer.get_slice_pixmap(slice_data, z)
            self._update_scene_pixmap(self.scene, pixmap, is_mip=False)
        
        # Render MIP
        layers = self.sliderMIP.value()
        mip_data = self.data_manager.compute_mip(z, layers)
        if mip_data is not None:
            pixmap = self.renderer.get_mip_pixmap(mip_data, z, layers)
            self._update_scene_pixmap(self.scene_mip, pixmap, is_mip=True)
        
        # Fit views
        self.graphicsView.fitInView(
            0, 0, IMAGE_SIZE, IMAGE_SIZE, Qt.KeepAspectRatio
        )
        self.graphicsView_MIP.fitInView(
            0, 0, IMAGE_SIZE, IMAGE_SIZE, Qt.KeepAspectRatio
        )
        
        # Update metrics and histogram
        self._update_metrics()
        self.histogram_timer.start(HISTOGRAM_UPDATE_DELAY_MS)
    
    def _update_scene_pixmap(
        self, 
        scene: QGraphicsScene, 
        pixmap: QPixmap, 
        is_mip: bool
    ):
        """Update pixmap in scene."""
        item = self._current_mip_item if is_mip else self._current_slice_item
        
        if item:
            item.setPixmap(pixmap)
        else:
            item = scene.addPixmap(pixmap)
            item.setZValue(-1)
            if is_mip:
                self._current_mip_item = item
            else:
                self._current_slice_item = item
        
        scene.setSceneRect(0, 0, IMAGE_SIZE, IMAGE_SIZE)
    
    def _on_circles_changed(self):
        """Handle circle position changes."""
        self._update_metrics()
        self.histogram_timer.start(HISTOGRAM_UPDATE_DELAY_MS)
    
    def _get_current_mip_data(self) -> Optional[np.ndarray]:
        """Get MIP data for current position."""
        if self.data_manager.data is None:
            return None
        
        z = self.sliderZ.value()
        layers = self.sliderMIP.value()
        return self.data_manager.compute_mip(z, layers)
    
    def _update_metrics(self):
        """Update metrics table."""
        mip_data = self._get_current_mip_data()
        if mip_data is None:
            return
        
        # Get data for both circles
        metrics_dict = {}
        positions = {}
        
        for circle_type in CircleType:
            x, y = self.circle_manager.get_position(circle_type)
            radius = self.circle_manager.get_radius(circle_type)
            
            data = self.analytics.get_circle_data(x, y, radius, mip_data)
            metrics_dict[circle_type] = Metrics.from_data(data)
            positions[circle_type] = (x, y)
        
        # Update table
        self.metrics_table.update(
            positions[CircleType.ASJ],
            positions[CircleType.ED],
            metrics_dict[CircleType.ASJ],
            metrics_dict[CircleType.ED]
        )
    
    def _update_histogram(self):
        """Update histogram display."""
        mip_data = self._get_current_mip_data()
        if mip_data is None:
            return
        
        # Get data for both circles
        data_arrays = {}
        for circle_type in CircleType:
            x, y = self.circle_manager.get_position(circle_type)
            radius = self.circle_manager.get_radius(circle_type)
            data_arrays[circle_type] = self.analytics.get_circle_data(
                x, y, radius, mip_data
            )
        
        self.histogram_canvas.plot_histograms(
            data_arrays[CircleType.ASJ],
            data_arrays[CircleType.ED]
        )
    
    def resizeEvent(self, event):
        """Handle window resize."""
        if self.data_manager.data is not None:
            self.graphicsView.fitInView(
                self.scene.sceneRect(), Qt.KeepAspectRatio
            )
            self.graphicsView_MIP.fitInView(
                self.scene_mip.sceneRect(), Qt.KeepAspectRatio
            )
        super().resizeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Viewer()
    window.show()
    sys.exit(app.exec_())
