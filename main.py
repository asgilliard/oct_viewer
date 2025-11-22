"""
OCT image viewer with MVC architecture.

asgilliard
"""
import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import darkdetect
import numba as nb
import numpy as np
import pyqtgraph as pg
from matplotlib import colormaps
from PySide2.QtCore import QDir, QPointF, Qt, QTimer
from PySide2.QtGui import QBrush, QColor, QImage, QPen, QPixmap
from PySide2.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QFileSystemModel,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QMainWindow,
    QTableWidgetItem,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from design_ui import Ui_MainWindow
from smart_graphics import SmartGraphicsView

# Constants
IMAGE_SIZE = 512
DEFAULT_Z = 256
DEFAULT_MIP_LAYERS = 16
DEFAULT_CIRCLE_RADIUS = 30
MAX_PIXMAP_CACHE_SIZE = 50
THEME_CHECK_DELAY_MS = 1000
MEMMAP_MODE = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CircleType(Enum):
    """Types of measurement circles."""

    ASJ = 0
    ED = 1


@dataclass
class Metrics:
    """Metrics calculated from image data."""

    max_value: int = 0
    median: float = 0.0
    q75_diff: float = 0.0
    variance: float = 0.0
    size: int = 0

    @classmethod
    def from_data(cls, data: np.ndarray) -> 'Metrics':
        """Fabric: calculates metrics from data array."""
        if data.size == 0:
            return cls()

        return cls(
            max_value=int(np.max(data)),
            median=float(np.median(data)),
            q75_diff=float(np.percentile(data, 75) - np.median(data)),
            variance=float(np.std(data))**2,
            size=int(data.size)
        )


@dataclass
class Coefficients:
    """Condition and Presence coefficients."""
    econd: float = 0.0  # Condition coefficient
    epres: float = 0.0  # Presence coefficient
    
    @classmethod
    def from_data(cls, data_asj: np.ndarray, data_ed: np.ndarray, q75_diff: float) -> 'Coefficients':
        """Calculate coefficients from ASJ and ED data.
        
        Args:
            data_asj: ASJ circle data
            data_ed: ED circle data
            q75_diff: Q75 difference (used as normalizer)
        """
        if data_asj.size == 0 or data_ed.size == 0 or q75_diff == 0:
            return cls()
        
        # Econd: abs(median difference) / q75_diff
        median_asj = np.median(data_asj)
        median_ed = np.median(data_ed)
        econd = float(abs(median_ed - median_asj) / q75_diff)
        
        # Epres: bright scatterer detection (10th brightest pixel method)
        if data_asj.size >= 10 and data_ed.size >= 10:
            top10_asj = float(np.partition(data_asj, -10)[-10])  # top 10 on brightness
            top10_ed = float(np.partition(data_ed, -10)[-10])
            epres = abs(top10_ed - top10_asj) / q75_diff
        else:
            # Fallback for small regions
            epres = float(abs((np.max(data_ed) - np.max(data_asj)) / q75_diff))
        
        return cls(econd=econd, epres=epres)


class ImageDataManager:
    """Manages 3D image data loading and access."""
    
    def __init__(self, use_memmap: bool = False, cache_mip: bool = False):
        """Initialize data manager.
        
        Args:
            use_memmap: If True, use memory-mapped file (for large files).
                       If False, load entire file into RAM (default, faster).
            cache_mip: If True, cache computed MIP slices (useful with memmap).
                      If False, compute MIP on-the-fly (default, saves memory).
        """
        self.data: Optional[np.ndarray] = None
        self._memmap: Optional[np.memmap] = None
        self._use_memmap = use_memmap
        self._cache_mip = cache_mip
        self._mip_cache: Dict[Tuple[int, int], np.ndarray] = {}
        
        self.depth: int = 0
        self.height: int = 0
        self.width: int = 0
        
    def load(self, file_path: Path) -> bool:
        """Load image data from file with auto-detection of dimensions."""
        self.clear()
        
        try:
            # Определяем размер файла
            file_size = file_path.stat().st_size
            logger.info(f"File size: {file_size} bytes ({file_size / (1024**2):.2f} MB)")
            
            # Определяем форму по размеру файла
            shape = self._detect_shape(file_size)
            if shape is None:
                logger.error(f"Unsupported file size: {file_size} bytes")
                return False
            
            self.depth, self.height, self.width = shape
            logger.info(f"Detected shape: {self.depth}×{self.height}×{self.width}")
            
            # Проверяем расчёт
            expected_size = self.depth * self.height * self.width
            logger.info(f"Expected size: {expected_size}, actual: {file_size}, match: {expected_size == file_size}")
            
            if self._use_memmap:
                self._memmap = np.memmap(
                    str(file_path), 
                    dtype=np.uint8, 
                    mode='r', 
                    shape=(self.depth, self.height, self.width)  # явно указываем порядок
                )
                self.data = self._memmap
                logger.info(f"Loaded as memmap: {file_path}")
            else:
                with open(file_path, 'rb') as f:
                    self.data = np.frombuffer(
                        f.read(),
                        dtype=np.uint8
                    ).reshape(self.depth, self.height, self.width)
                logger.info(f"Loaded into RAM: {file_path}")
            
            if self._cache_mip:
                self._mip_cache.clear()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load file: {e}", exc_info=True)  # покажет полный traceback
            return False
    
    @staticmethod
    def _detect_shape(file_size: int) -> Optional[Tuple[int, int, int]]:
        """Detect volume shape from file size."""
        known_shapes = [
            (512, 512, 512),  # 134,217,728 bytes
            (512, 256, 512),  # 67,108,864 bytes
        ]
            
        for shape in known_shapes:
            expected_size = shape[0] * shape[1] * shape[2]
            if file_size == expected_size:
                return shape
            
        return None
    
    def get_slice(self, z: int) -> Optional[np.ndarray]:
        """Get 2D slice at z position."""
        if self.data is None or not self._is_valid_z(z):
            return None
        return self.data[z, :, :]
    
    def compute_mip(self, z: int, layers: int) -> Optional[np.ndarray]:
        """Compute Maximum Intensity Projection.
        
        Args:
            z: Center z-coordinate
            layers: Number of layers to project
            
        Returns:
            2D array with maximum intensity projection, or None if data unavailable
        """
        if self.data is None or not self._is_valid_z(z):
            return None
        
        # Use cache if enabled
        if self._cache_mip:
            key = (z, layers)
            if key not in self._mip_cache:
                self._mip_cache[key] = self._compute_mip_data(z, layers)
            return self._mip_cache[key]
        
        # Otherwise compute on-the-fly
        return self._compute_mip_data(z, layers)
    
    def _compute_mip_data(self, z: int, layers: int) -> np.ndarray:
        """Internal MIP computation."""
        if self.data is None:
            return np.array([])
        
        half_layers = layers // 2
        z_min = max(0, z - half_layers)
        z_max = min(self.depth, z + (layers - half_layers))
        return np.max(self.data[z_min:z_max, :, :], axis=0)
    
    def _is_valid_z(self, z: int) -> bool:
        """Check if z coordinate is valid."""
        return 0 <= z < self.depth
    
    def clear(self):
        """Clear all data and caches."""
        if self._memmap is not None:
            try:
                del self._memmap
            except Exception:
                pass
            finally:
                self._memmap = None
            
        self.data = None
        self.depth = 0
        self.height = 0
        self.width = 0
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
            self._cache.popitem(last=False)  # evict oldest
        self._cache[key] = pixmap

    def clear(self):
        """Clear cache."""
        self._cache.clear()

    def remove_by_condition(self, condition):
        """Remove items matching condition."""
        keys_to_remove = [k for k in self._cache.keys() if condition(k)]
        for k in keys_to_remove:
            del self._cache[k]


class Analytics:
    """Analytical calculations on image data."""

    def __init__(self):
        self._mask_cache: Dict[int, np.ndarray] = {}

    def get_circle_data(self, x: int, y: int, radius: int, slice_data: np.ndarray) -> np.ndarray:
        """Extract data inside circle region."""
        height, width = slice_data.shape
        
        if not (0 <= x < width and 0 <= y < height):
            return np.array([])

        if radius not in self._mask_cache:
            self._mask_cache[radius] = self._create_circle_mask(radius)

        mask = self._mask_cache[radius]

        x_min = max(0, x - radius)
        x_max = min(width, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(height, y + radius + 1)

        region = slice_data[y_min:y_max, x_min:x_max]

        if region.size == 0:
            return np.array([])

        mask_cropped = mask[: region.shape[0], : region.shape[1]]

        return region[mask_cropped]
        
    def get_circle_data_volume(
        self, 
        x: int, 
        y: int, 
        radius: int,
        z: int,
        layers: int,
        data_3d: np.ndarray
    ) -> np.ndarray:
        """Extract data inside circle region across multiple Z slices (volume)."""
        depth, height, width = data_3d.shape
        
        if not (0 <= x < width and 0 <= y < height):
            return np.array([])
        
        # Z range
        half1 = layers // 2
        half2 = layers - half1
        z_min = max(0, z - half1)
        z_max = min(depth, z + half2)
        
        # Get circle mask
        if radius not in self._mask_cache:
            self._mask_cache[radius] = self._create_circle_mask(radius)
        mask = self._mask_cache[radius]
        
        # Region borders XY
        x_min = max(0, x - radius)
        x_max = min(width, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(height, y + radius + 1)
        
        # Get all slices data
        volume_data = []
        for z_slice in range(z_min, z_max):
            slice_data = data_3d[z_slice, y_min:y_max, x_min:x_max]
            if slice_data.size == 0:
                continue
            
            mask_cropped = mask[:slice_data.shape[0], :slice_data.shape[1]]
            volume_data.append(slice_data[mask_cropped])
        
        return np.concatenate(volume_data) if volume_data else np.array([])
        
    @staticmethod
    def _create_circle_mask(radius: int) -> np.ndarray:
        """Create circular mask."""
        y_grid, x_grid = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        return x_grid**2 + y_grid**2 <= radius**2

    def clear_cache(self):
        """Clear mask cache."""
        self._mask_cache.clear()


class PyQtGraphHistogram(QWidget):
    def __init__(self, title: str = ""):
        super().__init__()
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Checkbox
        self.crosshair_checkbox = QCheckBox("Crosshair Mode")
        self.crosshair_checkbox.stateChanged.connect(self.toggle_crosshair_mode)
        layout.addWidget(self.crosshair_checkbox)
        
        # Plot
        self.plot_widget = pg.PlotWidget()
        self.update_background()
        layout.addWidget(self.plot_widget)
        
        if self.plot_widget.plotItem is None:
            return
        
        self.curve1 = self.plot_widget.plot(pen=pg.mkPen('r', width=1), fillLevel=0, brush=(255,0,0,100))
        self.curve2 = self.plot_widget.plot(pen=pg.mkPen('g', width=1), fillLevel=0, brush=(0,255,0,100))
        
        self._original_title = title
        if title:
            self.plot_widget.plotItem.setTitle(title)
                
        self.plot_widget.setLabel('bottom', 'Intensity')
        self.plot_widget.setLabel('left', 'Count')
        
        # Crosshair setup
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False)
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False)
        self.crosshair_v.setPen(pg.mkPen('y', width=1, style=Qt.DashLine))
        self.crosshair_h.setPen(pg.mkPen('y', width=1, style=Qt.DashLine))
        
        self.plot_widget.plotItem.addItem(self.crosshair_v)
        self.plot_widget.plotItem.addItem(self.crosshair_h)
        
        self.crosshair_v.hide()
        self.crosshair_h.hide()
        
        self._crosshair_mode = False
        self._mouse_pressed = False
        
        # Proxy for mouse
        self.proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved, 
            rateLimit=60, 
            slot=self.update_crosshair
        )
        
        # Mouse events
        self.plot_widget.scene().sigMouseClicked.connect(self.on_mouse_click)
        
        self.setMinimumSize(50, 50)
    
    def update_background(self):
        """Update background (on theme switching)"""
        self.plot_widget.setBackground(background=None)
        
    def toggle_crosshair_mode(self, state):
        if self.plot_widget.plotItem is None:
            return
        
        self._crosshair_mode = bool(state)
        view_box = self.plot_widget.plotItem.getViewBox()
        
        if view_box is None:
            return
        
        if self._crosshair_mode:
            # Disable mouse
            view_box.setMouseEnabled(x=False, y=False)
        else:
            # Enable back
            view_box.setMouseEnabled(x=True, y=True)
            self.crosshair_v.hide()
            self.crosshair_h.hide()
            self._mouse_pressed = False
            self.plot_widget.plotItem.setTitle(self._original_title)
    
    def on_mouse_click(self, event):
        if not self._crosshair_mode:
            return
            
        if event.button() == Qt.LeftButton:
            if event.double():
                return
            self._mouse_pressed = not self._mouse_pressed
            
            if self.plot_widget.plotItem is None:
                return
            
            if self._mouse_pressed:
                self.crosshair_v.show()
                self.crosshair_h.show()
            else:
                self.crosshair_v.hide()
                self.crosshair_h.hide()
                self.plot_widget.plotItem.setTitle(self._original_title)
        
    def update_crosshair(self, event):
        if not self._crosshair_mode or not self._mouse_pressed:
            return
            
        pos = event[0]
        
        if self.plot_widget.plotItem is None:
            return
        
        if self.plot_widget.plotItem.sceneBoundingRect().contains(pos):
            view_box = self.plot_widget.plotItem.getViewBox()
            
            if view_box is None:
                return
                
            mouse_point = view_box.mapSceneToView(pos)
            
            x = mouse_point.x()
            y = mouse_point.y()
            
            # Borders based on data
            x_min, x_max = 0, 255
            if self.curve1.yData is not None:
                y_max = max(self.curve1.yData.max(), self.curve2.yData.max()) if self.curve2.yData is not None else self.curve1.yData.max()
                y = max(0, min(y, y_max))
            
            x = max(x_min, min(x, x_max))
            x_int = int(x)
            
            self.crosshair_v.setPos(x)
            self.crosshair_h.setPos(y)
            
            if (self.curve1.yData is not None and 
                self.curve2.yData is not None and 
                0 <= x_int < len(self.curve1.yData)):
                
                y1 = int(self.curve1.yData[x_int])
                y2 = int(self.curve2.yData[x_int])
                
                self.plot_widget.plotItem.setTitle(
                    f"<span style='font-size: 10pt'>{self._original_title} | "
                    f"Int: {x_int} | R: {y1}, G: {y2}</span>"
                )
            else:
                self.plot_widget.plotItem.setTitle(
                    f"<span style='font-size: 10pt'>{self._original_title} | "
                    f"Int: {x:.0f}, Count: {y:.0f}</span>"
                )
        
    def update_data(self, data1, data2):
        data1_filtered = data1[data1 > 0]
        data2_filtered = data2[data2 > 0]
                
        if data1_filtered.size == 0 or data2_filtered.size == 0:
            return
        
        h1, edges = np.histogram(data1_filtered, bins=256, range=(0, 256))
        h2, _ = np.histogram(data2_filtered, bins=256, range=(0, 256))
        
        self.curve1.setData(edges[:-1], h1)
        self.curve2.setData(edges[:-1], h2)


class DraggableCircle(QGraphicsEllipseItem):
    """Draggable circle with position change signals."""

    def __init__(
        self,
        x: float,
        y: float,
        radius: int = DEFAULT_CIRCLE_RADIUS,
        color: Optional[QColor] = None,
        label: str = ""
    ):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.setPos(x, y)
        self.base_radius = radius
        self.radius = radius

        self.setBrush(QBrush())
        actual_color = color if color is not None else QColor(Qt.red)
        self.setPen(QPen(actual_color, 2))

        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)

        self._is_syncing = False
        self._position_callback = None
        self._radius_callback = None
        
        if label:
            self.label_item = QGraphicsTextItem(label, self)
            self.label_item.setDefaultTextColor(actual_color)

            font = self.label_item.font()
            font.setPointSize(14)
            font.setBold(True)
            self.label_item.setFont(font)

            text_rect = self.label_item.boundingRect()
            self.label_item.setPos(
                -text_rect.width() / 2,
                -text_rect.height() / 2
            )
                    
            self.label_item.setFlag(QGraphicsItem.ItemIgnoresTransformations)

    def set_position_callback(self, callback):
        """Set callback for position changes."""
        self._position_callback = callback
        
    def set_radius_callback(self, callback):
        """Set callback for radius changes."""
        self._radius_callback = callback

    def itemChange(self, change, value):
        """Handle item changes with proper zoom-aware bounds checking."""
        if change == QGraphicsItem.ItemPositionChange and not self._is_syncing:
            if not self.scene():
                return super().itemChange(change, value)
            
            scene_rect = self.scene().sceneRect()
            
            # Get the current view for calculating the effective boundaries
            view = self.scene().views()[0] if self.scene().views() else None
            if not view:
                return super().itemChange(change, value)
            
            view_rect = view.mapToScene(view.viewport().rect()).boundingRect()

            current_radius = self.radius
            
            # Limit the position to the visible area of the view
            bounded_x = max(view_rect.left() + current_radius, 
                           min(view_rect.right() - current_radius, value.x()))
            bounded_y = max(view_rect.top() + current_radius, 
                           min(view_rect.bottom() - current_radius, value.y()))
            
            # Additionally, limit the radius of the circles to the boundaries of the scene
            bounded_x = max(scene_rect.left() + current_radius, 
                           min(scene_rect.right() - current_radius, bounded_x))
            bounded_y = max(scene_rect.top() + current_radius, 
                           min(scene_rect.bottom() - current_radius, bounded_y))
            
            new_pos = QPointF(bounded_x, bounded_y)
            
            if self._position_callback:
                self._position_callback()
            
            return new_pos
            
        elif change == QGraphicsItem.ItemTransformChange and not self._is_syncing:
            return value
        
        return super().itemChange(change, value)
        
    def sync_position(self, pos: QPointF):
        """Sync position without triggering callbacks."""
        self._is_syncing = True
        self.setPos(pos)
        self._is_syncing = False
        
    def sync_radius(self, radius: int):
        """Sync position without triggering callbacks."""
        if radius == self.radius:
            return
        self._is_syncing = True
        self.radius = radius
        self.setRect(-radius, -radius, radius * 2, radius * 2)
        if hasattr(self, 'label_item'):
            text_rect = self.label_item.boundingRect()
            self.label_item.setPos(-text_rect.width() / 2, -text_rect.height() / 2)
        self._is_syncing = False

class CircleManager:
    """Manages pairs of synchronized circles."""

    def __init__(self):
        self.circles: Dict[CircleType, DraggableCircle] = {}
        self.circles_mip: Dict[CircleType, DraggableCircle] = {}
        self._sync_callback = None
        self._current_scale = 1.0

    def create_circles(self, scene: QGraphicsScene, scene_mip: QGraphicsScene):
        """Create circle pairs for both scenes."""
        colors = {CircleType.ASJ: QColor(255, 0, 0), CircleType.ED: QColor(0, 255, 0)}
        labels = {CircleType.ASJ: "ASJ", CircleType.ED: "ED"}

        for i, circle_type in enumerate(CircleType):
            x_pos = 100 + i * 100

            # Main scene circle
            circle = DraggableCircle(x_pos, 100, color=colors[circle_type], label=labels[circle_type])
            circle.set_position_callback(
                lambda ct=circle_type: self._on_position_changed(ct, False)
            )
            self.circles[circle_type] = circle
            scene.addItem(circle)

            # MIP scene circle
            circle_mip = DraggableCircle(x_pos, 100, color=colors[circle_type], label=labels[circle_type])
            circle_mip.set_position_callback(
                lambda ct=circle_type: self._on_position_changed(ct, True)
            )
            self.circles_mip[circle_type] = circle_mip
            scene_mip.addItem(circle_mip)

    def set_callbacks(self, sync_callback):
        """Set all callbacks."""
        self._sync_callback = sync_callback

    # def _on_position_changed(self, circle_type: CircleType, is_mip: bool):
    #     """Handle position change of a circle."""
    #     source = self.circles_mip[circle_type] if is_mip else self.circles[circle_type]
    #     target = self.circles[circle_type] if is_mip else self.circles_mip[circle_type]

    #     target.sync_position(source.pos())

    #     if self._sync_callback:
    #         self._sync_callback()
    # 
    def _on_position_changed(self, circle_type: CircleType, is_mip: bool):
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
        return int(self.circles[circle_type].radius)

    def update_scale(self, scale: float):
        if abs(scale - self._current_scale) < 0.001:
            return
        self._current_scale = scale
        new_radius = max(5, int(DEFAULT_CIRCLE_RADIUS / scale))
    
        for circle_type in CircleType:
            self.circles[circle_type].sync_radius(new_radius)
            self.circles_mip[circle_type].sync_radius(new_radius)
    
        if self._sync_callback:
            self._sync_callback()

class MetricsTable:
    """Manages metrics display table."""

    METRIC_LABELS = ['X', 'Y', 'Max', 'Med', 'Q75-med', 'Var', 'Size']

    def __init__(self, table_widget):
        self.table = table_widget
        self._setup_table()

    def _setup_table(self):
        """Initialize table structure."""
        self.table.setRowCount(len(self.METRIC_LABELS))
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Metric", "ASJ_M", "ED_M", "ASJ_V", "ED_V"
        ])

        for i, label in enumerate(self.METRIC_LABELS):
            self.table.setItem(i, 0, QTableWidgetItem(label))

    def update(
        self, 
        pos_asj: Tuple[int, int], 
        pos_ed: Tuple[int, int],
        metrics_asj_mip: Metrics,
        metrics_ed_mip: Metrics,
        metrics_asj_vol: Metrics,
        metrics_ed_vol: Metrics
    ):
        """Update table with both MIP and Volume metrics."""
        # Positions (same)
        self.table.setItem(0, 1, QTableWidgetItem(str(pos_asj[0])))
        self.table.setItem(1, 1, QTableWidgetItem(str(pos_asj[1])))
        self.table.setItem(0, 2, QTableWidgetItem(str(pos_ed[0])))
        self.table.setItem(1, 2, QTableWidgetItem(str(pos_ed[1])))
            
        # MIP
        self._update_column(1, metrics_asj_mip, skip_pos=True)
        self._update_column(2, metrics_ed_mip, skip_pos=True)
            
        # Volume
        self._update_column(3, metrics_asj_vol, skip_pos=True)
        self._update_column(4, metrics_ed_vol, skip_pos=True)
        
    def _update_column(self, col: int, metrics: Metrics, skip_pos: bool = False):
        """Update single column with metrics."""
        start_row = 2 if skip_pos else 0
            
        if not skip_pos:
            # X, Y are filled yet
            pass
            
        self.table.setItem(start_row + 0, col, QTableWidgetItem(str(metrics.max_value)))
        self.table.setItem(start_row + 1, col, QTableWidgetItem(f"{metrics.median:.1f}"))
        self.table.setItem(start_row + 2, col, QTableWidgetItem(f"{metrics.q75_diff:.1f}"))
        self.table.setItem(start_row + 3, col, QTableWidgetItem(f"{metrics.variance:.1f}"))
        self.table.setItem(start_row + 4, col, QTableWidgetItem(str(metrics.size)))
        
        
class ImageRenderer:
    """Image renderer with caching"""
    
    def __init__(self):
        self._pixmap_cache = PixmapCache()
        self._mip_pixmap_cache = PixmapCache()
        self._cmap_cache = {}
        self._lut_cache = {}
        self._numpy_refs = []
    
    def _build_color_lut(self, cmap_name: str, invert: bool = False) -> np.ndarray:
        """Build color lookup table"""
        cache_key = (cmap_name, invert)
        if cache_key in self._lut_cache:
            return self._lut_cache[cache_key]
    
        if cmap_name not in self._cmap_cache:
            self._cmap_cache[cmap_name] = colormaps.get_cmap(cmap_name)
    
        cmap = self._cmap_cache[cmap_name]
        indices = np.arange(256, dtype=np.float32) / 255.0
        if invert:
            indices = 1.0 - indices
    
        rgba = cmap(indices)  # (256, 4) float
        rgb = (rgba[:, :3] * 255).astype(np.uint8)
    
        self._lut_cache[cache_key] = rgb
        return rgb
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _apply_lut_numba(arr: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """Apply color lookup table with Numba"""
        height, width = arr.shape
        rgb = np.empty((height, width, 3), dtype=np.uint8)
            
        for i in nb.prange(height):
            for j in range(width):
                idx = arr[i, j]
                rgb[i, j, 0] = lut[idx, 0]  # R
                rgb[i, j, 1] = lut[idx, 1]  # G
                rgb[i, j, 2] = lut[idx, 2]  # B
            
        return rgb
    
    def array_to_qimage(self, arr: np.ndarray, invert: bool = False, cmap: str = 'gray') -> QImage:
        """Convert numpy array to QImage - Numba optimized"""
        if arr.size == 0:
            return QImage()
    
        height, width = arr.shape
    
        # Fast path for grayscale
        if cmap == 'gray':
            if invert:
                arr = self._invert_array(arr)
            else:
                arr = np.copy(arr)
                    
            if not arr.flags['C_CONTIGUOUS']:
                    arr = np.ascontiguousarray(arr)
                    
            bytes_per_line = width
            qimg = QImage(arr.data, width, height, bytes_per_line, QImage.Format_Grayscale8)  # type: ignore
            self._numpy_refs.append(arr)
                
            if len(self._numpy_refs) > 10:
                self._numpy_refs = self._numpy_refs[-10:]
                    
            return qimg
    
        # Color maps with Numba
        lut = self._build_color_lut(cmap, invert)
        rgb_data = self._apply_lut_numba(arr, lut)
    
        bytes_per_line = 3 * width
        qimg = QImage(rgb_data.data, width, height, bytes_per_line, QImage.Format_RGB888)  # type: ignore
        self._numpy_refs.append(rgb_data)
            
        if len(self._numpy_refs) > 10:
            self._numpy_refs = self._numpy_refs[-10:]
                
        return qimg
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _invert_array(arr: np.ndarray) -> np.ndarray:
        """Invert array values with Numba"""
        result = np.empty_like(arr)
        for i in nb.prange(arr.shape[0]):
            for j in range(arr.shape[1]):
                result[i, j] = 255 - arr[i, j]
        return result
    
    def get_slice_pixmap(
        self, slice_data: np.ndarray, z: int, theme: str, palette: str = 'gray'
    ) -> QPixmap:
        """Get pixmap for slice with caching"""
        cache_key = (z, theme, palette)

        cached = self._pixmap_cache.get(cache_key)
        if cached:
            return cached

        invert = theme == 'Light'
        qimage = self.array_to_qimage(slice_data, invert=invert, cmap=palette)
        pixmap = QPixmap.fromImage(qimage)
        self._pixmap_cache.put(cache_key, pixmap)
        return pixmap

    def get_mip_pixmap(
        self, mip_data: np.ndarray, z: int, layers: int, theme: str, palette: str = 'gray'
    ) -> QPixmap:
        """Get pixmap for MIP with caching"""
        cache_key = (z, layers, theme, palette)

        cached = self._mip_pixmap_cache.get(cache_key)
        if cached:
            return cached

        invert = theme == 'Light'
        qimage = self.array_to_qimage(mip_data, invert=invert, cmap=palette)
        pixmap = QPixmap.fromImage(qimage)
        self._mip_pixmap_cache.put(cache_key, pixmap)
        return pixmap

    def cleanup_caches(self, current_z: int, current_layers: int):
        """Clean up old cache entries"""
        self._mip_pixmap_cache.remove_by_condition(
            lambda k: isinstance(k, tuple) and k[1] != current_layers
        )

    def clear(self):
        """Clear all caches"""
        self._pixmap_cache.clear()
        self._mip_pixmap_cache.clear()      


class Viewer(QMainWindow, Ui_MainWindow):
    """Main viewer window."""

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self._circle_data_cache = {}
        self._current_folder: Optional[Path] = None
        self._file_list: list[Path] = []
        self._current_file_index: int = -1 
        
        self._file_states: Dict[Path, dict] = {}

        # Flags
        self._current_theme = self.get_system_theme()
        self._current_palette: str = 'gray'

        # Components
        if MEMMAP_MODE:
            self.data_manager = ImageDataManager(use_memmap=True, cache_mip=True)
        else:
            self.data_manager = ImageDataManager()
        self.analytics = Analytics()
        self.renderer = ImageRenderer()
        self.circle_manager = CircleManager()

        # UI components
        self._current_slice_item = None
        self._current_mip_item = None

        # Setup
        self._setup_file_tree()
        self._setup_palette_menu()
        self._setup_scenes()
        self._setup_sliders()
        self._setup_theme_timer()
        self._setup_histogram()
        self._setup_metrics_table()
        self._setup_circles()
        self._setup_connections()
        self._warmup_numba()

    @staticmethod
    def get_system_theme() -> str:
        """Detect system theme (light/dark)."""
        try:
            theme = darkdetect.theme()
            return theme if theme is not None else 'Light'
        except Exception:
            return 'Light'

    def _check_theme(self):
        """Check for theme changes."""
        new_theme = self.get_system_theme()
        if new_theme != self._current_theme:
            self._current_theme = new_theme
            self.renderer.clear()
            self.histogram_canvas_mip.update_background()
            self.histogram_canvas_volume.update_background()
            self.update_views(self.sliderZ.value(), update_histogram=False)
    
    def _setup_file_tree(self):
        """Initialize file tree dock."""
        self.file_dock = QDockWidget("Files", self)
        self.file_tree = QTreeView()
        
        self.file_model = QFileSystemModel()
        self.file_model.setFilter(QDir.Files | QDir.NoDotAndDotDot)
        self.file_model.setNameFilters(["*.dat"])
        self.file_model.setNameFilterDisables(False)
        
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(""))
        
        # Hide all columns except name
        for i in range(1, 4):
            self.file_tree.hideColumn(i)
        
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setIndentation(0)
        self.file_tree.setUniformRowHeights(True)           # same row height
        self.file_tree.setAlternatingRowColors(True)        # row colors
        
        self.file_tree.clicked.connect(self._on_file_tree_clicked)
        self.file_tree.doubleClicked.connect(self._on_file_tree_clicked)  # + double click
        
        self.file_dock.setWidget(self.file_tree)
        self.file_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_dock)
        self.file_dock.hide()
    
    def _setup_scenes(self):
        """Initialize scenes."""
        self.scene = QGraphicsScene(self)
        self.scene_mip = QGraphicsScene(self)
        
        self.graphicsView: SmartGraphicsView
        self.graphicsView_MIP: SmartGraphicsView
        
        self.graphicsView.setScene(self.scene)
        self.graphicsView_MIP.setScene(self.scene_mip)
        
        self.graphicsView.zoomChanged.connect(self.graphicsView_MIP.sync_zoom)
        self.graphicsView_MIP.zoomChanged.connect(self.graphicsView.sync_zoom)
        
        self.graphicsView.panChanged.connect(self.graphicsView_MIP.sync_pan)
        self.graphicsView_MIP.panChanged.connect(self.graphicsView.sync_pan)
        
        self.graphicsView.centerCirclesRequested.connect(self.center_circles_on_view)
        self.graphicsView_MIP.centerCirclesRequested.connect(self.center_circles_on_view)
        
        self.graphicsView.scaleChanged.connect(self.circle_manager.update_scale)
        self.graphicsView_MIP.scaleChanged.connect(self.circle_manager.update_scale)
        
        self.scene.setSceneRect(0, 0, IMAGE_SIZE, IMAGE_SIZE)
        self.scene_mip.setSceneRect(0, 0, IMAGE_SIZE, IMAGE_SIZE)

    def _setup_sliders(self):
        """Initialize sliders."""
        self.sliderZ.setRange(0, IMAGE_SIZE - 1)
        self.sliderZ.setValue(DEFAULT_Z)
        self.sliderMIP.setRange(2, 100)
        self.sliderMIP.setValue(DEFAULT_MIP_LAYERS)

        self.current_z_label.setText(f'Z: {DEFAULT_Z}')
        self.MIP_layers_label.setText(f'MIP: {DEFAULT_MIP_LAYERS}')

    def _setup_histogram(self):
        """Initialize histogram canvas."""
        # MIP histogram
        self.histogram_canvas_mip = PyQtGraphHistogram(title="MIP")
        histogram_layout = QVBoxLayout(self.histogramWidget_MIP)
        histogram_layout.addWidget(self.histogram_canvas_mip)
            
        # Volume histogram
        self.histogram_canvas_volume = PyQtGraphHistogram(title="Volume")
        histogram_volume_layout = QVBoxLayout(self.histogramWidget_Volume)
        histogram_volume_layout.addWidget(self.histogram_canvas_volume)

    def _setup_metrics_table(self):
        """Initialize metrics table."""
        self.metrics_table = MetricsTable(self.tableMetrics)

    def _setup_circles(self):
        """Initialize measurement circles."""
        self.circle_manager.create_circles(self.scene, self.scene_mip)
        self.circle_manager.set_callbacks(self._on_circles_changed)
        
    def center_circles_on_view(self):
        """Move circles to the current view center."""
        center = self.graphicsView.mapToScene(self.graphicsView.viewport().rect().center())
        
        offset = 50  # distance between circles
        
        for i, circle_type in enumerate(CircleType):
            x = center.x() + (i - 0.5) * offset
            y = center.y()
            pos = QPointF(x, y)
            
            self.circle_manager.circles[circle_type].sync_position(pos)
            self.circle_manager.circles_mip[circle_type].sync_position(pos)
            
        current_scale = self.graphicsView.transform().m11()
        self.circle_manager.update_scale(current_scale)
    
        self._circle_data_cache.clear()
        self._update_metrics()
        self._update_histogram()

    def _setup_palette_menu(self):
        self._palette_actions: dict[QAction, str] = {
            self.actionGray: 'gray',
            self.actionViridis: 'viridis',
            self.actionPlasma: 'plasma',
            self.actionMagma: 'magma',
            self.actionInferno: 'inferno',
            self.actionCividis: 'cividis',
            self.actionJet: 'jet',
        }

        for action, palette_name in self._palette_actions.items():
            action.setCheckable(True)
            action.triggered.connect(partial(self._on_palette_selected, palette_name))

        self.actionGray.setChecked(True)

    def _setup_connections(self):
        """Connect signals and slots."""
        self.actionOpen.triggered.connect(self.open_file)
        self.sliderZ.valueChanged.connect(self.on_z_changed)
        self.sliderMIP.valueChanged.connect(self.on_layers_changed)
        
        self.actionOpen_Folder.triggered.connect(self.open_folder)
        self.actionPrevious_File.triggered.connect(self.load_previous_file)
        self.actionNext_File.triggered.connect(self.load_next_file)
        
        self.actionShow_File_Tree.toggled.connect(self.file_dock.setVisible)
        self.file_dock.visibilityChanged.connect(self.actionShow_File_Tree.setChecked)
        
        self.pushButtonCenterCircles.clicked.connect(self.center_circles_on_view)

    def _on_palette_selected(self, palette_name: str):
        self._current_palette = palette_name
        for action, name in self._palette_actions.items():
            action.setChecked(name == palette_name)
        self.renderer.clear()
        self.update_views(self.sliderZ.value(), update_histogram=False)

    def _setup_theme_timer(self):
        """Initialize theme update timer."""
        self._current_theme = self.get_system_theme()
        self._theme_timer = QTimer()
        self._theme_timer.timeout.connect(self._check_theme)
        self._theme_timer.start(THEME_CHECK_DELAY_MS)

    def _warmup_numba(self):
            """Warming up numba JIT compilation"""
            logger.info("Warming up Numba JIT...")
            
            dummy = np.zeros((32, 32), dtype=np.uint8)
            dummy_lut = np.zeros((256, 3), dtype=np.uint8)
            
            self.renderer._invert_array(dummy)
            self.renderer._apply_lut_numba(dummy, dummy_lut)
            
            logger.info("Numba JIT ready!")
    
    def open_file(self):
        """Open and load image file."""
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open .dat', '', 'DAT Files (*.dat)')
        
        if not file_name:
            return
            
        file_path = Path(file_name)
            
        if self._current_folder != file_path.parent:
            self._current_folder = file_path.parent
            self._update_file_list()

            self.file_model.setRootPath(str(self._current_folder))
            self.file_tree.setRootIndex(self.file_model.index(str(self._current_folder)))
        
        self._load_file(Path(file_name))
        self.file_dock.hide()
    
    def _load_file(self, file_path: Path):
        """Load specific file."""
        logger.info(f"Loading file: {file_path}")
        
        # Save previous file state
        if self.data_manager.data is not None and hasattr(self, '_current_file_path'):
            self._file_states[self._current_file_path] = {
                'circle_positions': {
                    circle_type: self.circle_manager.get_position(circle_type)
                    for circle_type in CircleType
                },
                'transform': self.graphicsView.transform(),
                'center': self.graphicsView.mapToScene(self.graphicsView.viewport().rect().center()),
                'z': self.sliderZ.value(),
                'mip': self.sliderMIP.value(),
            }
        
        self._clear_all()
        self._current_file_path = file_path
        
        if not self.data_manager.load(file_path):
            logger.error(f"Failed to load file: {file_path}")
            return
        
        # Update file list index
        if self._current_folder is None:
            self._current_folder = file_path.parent
            self._update_file_list()
        
        if file_path in self._file_list:
            self._current_file_index = self._file_list.index(file_path)
        
        file_size = file_path.stat().st_size
        
        width = self.data_manager.width
        height = self.data_manager.height
        depth = self.data_manager.depth
        
        logger.info(f"Loaded: {depth}×{height}×{width}")
        
        # Restore Z and MIP or set default
        saved_state = self._file_states.get(file_path)
        if saved_state:
            default_z = saved_state.get('z', depth // 2)
            default_mip = saved_state.get('mip', DEFAULT_MIP_LAYERS)
        else:
            default_z = depth // 2
            default_mip = DEFAULT_MIP_LAYERS
            
        self.sliderZ.setRange(0, depth - 1)
        self.sliderZ.setValue(default_z)
        self.sliderMIP.setValue(default_mip)
    
        self.scene.setSceneRect(0, 0, width, height)
        self.scene_mip.setSceneRect(0, 0, width, height)
        
        self.update_views(default_z)
        
        QApplication.processEvents()
        
        # Restore zoom or fit
        if saved_state and 'transform' in saved_state:
            self.graphicsView.setTransform(saved_state['transform'])
            self.graphicsView_MIP.setTransform(saved_state['transform'])
            
            if 'center' in saved_state:
                self.graphicsView.centerOn(saved_state['center'])
                self.graphicsView_MIP.centerOn(saved_state['center'])
                
            self.circle_manager.update_scale(self.graphicsView.transform().m11())
            
        else:
            self.graphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self.graphicsView_MIP.fitInView(self.scene_mip.sceneRect(), Qt.KeepAspectRatio)
            self.circle_manager.update_scale(1.0)
        
        # Restore circle positions
        if saved_state and 'circle_positions' in saved_state:
            for circle_type, (x, y) in saved_state['circle_positions'].items():
                pos = QPointF(x, y)
                self.circle_manager.circles[circle_type].sync_position(pos)
                self.circle_manager.circles_mip[circle_type].sync_position(pos)
        
        # Force update metrics and hists
        self._circle_data_cache.clear()
        self._update_metrics()
        self._update_histogram()
        
        # Update status
        file_info = f"{file_path.name} | {depth}×{height}×{width} | {file_size / (1024**2):.1f} MB"
        if self._file_list:
            file_info += f" | File {self._current_file_index + 1}/{len(self._file_list)}"
            
        self.pathLabel.setText(file_info)
        logger.info("File loaded successfully")

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
        self.current_z_label.setText(f'Z: {z}')
        self._circle_data_cache.clear()
        self.update_views(z)

    def on_layers_changed(self, layers: int):
        """Handle MIP layers slider change."""
        self.MIP_layers_label.setText(f'MIP: {layers}')
        self._circle_data_cache.clear()
        self.renderer.cleanup_caches(self.sliderZ.value(), layers)
        self.update_views(self.sliderZ.value())

    def update_views(self, z: int, update_histogram: bool = True):
        """Update all views for given Z position."""
        
        # Render main slice
        slice_data = self.data_manager.get_slice(z)
        if slice_data is not None:
            pixmap = self.renderer.get_slice_pixmap(
                slice_data, z, self._current_theme, self._current_palette
            )
            self._update_scene_pixmap(self.scene, pixmap, is_mip=False)
    
        # Render MIP
        layers = self.sliderMIP.value()
        mip_data = self.data_manager.compute_mip(z, layers)
        if mip_data is not None:
            pixmap = self.renderer.get_mip_pixmap(
                mip_data, z, layers, self._current_theme, self._current_palette
            )
            self._update_scene_pixmap(self.scene_mip, pixmap, is_mip=True)
    
        # Update metrics and histogram
        self._update_metrics()
        if update_histogram:
            self._update_histogram()

    def _update_scene_pixmap(self, scene: QGraphicsScene, pixmap: QPixmap, is_mip: bool):
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

    def _on_circles_changed(self):
        """Handle circle position changes."""
        self._circle_data_cache.clear()
        self._update_metrics()
        self._update_histogram()
        
    def _get_circle_data_cached(self, circle_type: CircleType) -> tuple:
        """Get circle data from cache."""
        if self.data_manager.data is None:
            return None, None
                
        z = self.sliderZ.value()
        layers = self.sliderMIP.value()
        x, y = self.circle_manager.get_position(circle_type)
        radius = self.circle_manager.get_radius(circle_type)
            
        key = (circle_type, x, y, z, layers)
            
        if key not in self._circle_data_cache:
            mip_data = self._get_current_mip_data()
            if mip_data is None:
                return None, None
                    
            data_mip = self.analytics.get_circle_data(x, y, radius, mip_data)
            data_vol = self.analytics.get_circle_data_volume(
                x, y, radius, z, layers, self.data_manager.data
            )
            self._circle_data_cache[key] = (data_mip, data_vol)
            
        return self._circle_data_cache[key]

    def _get_current_mip_data(self) -> Optional[np.ndarray]:
        """Get MIP data for current position."""
        if self.data_manager.data is None:
            return None

        z = self.sliderZ.value()
        layers = self.sliderMIP.value()
        return self.data_manager.compute_mip(z, layers)
    
    def _update_metrics(self):
        """Update metrics table and coefficients."""
        if self.data_manager.data is None:
            return
        
        data_dict = {}
        
        for circle_type in CircleType:
            data_mip, data_vol = self._get_circle_data_cached(circle_type)
            if data_mip is None:
                return
            data_dict[circle_type] = (data_mip, data_vol)
    
        # Metrics
        metrics_asj_mip = Metrics.from_data(data_dict[CircleType.ASJ][0])
        metrics_ed_mip = Metrics.from_data(data_dict[CircleType.ED][0])
        metrics_asj_vol = Metrics.from_data(data_dict[CircleType.ASJ][1])
        metrics_ed_vol = Metrics.from_data(data_dict[CircleType.ED][1])
        
        self.metrics_table.update(
            self.circle_manager.get_position(CircleType.ASJ),
            self.circle_manager.get_position(CircleType.ED),
            metrics_asj_mip, metrics_ed_mip,
            metrics_asj_vol, metrics_ed_vol
        )
        
        # Coefficients for MIP
        coeff_mip = Coefficients.from_data(
            data_dict[CircleType.ASJ][0],
            data_dict[CircleType.ED][0],
            metrics_ed_mip.q75_diff
        )
        
        # Coefficients for Volume
        coeff_vol = Coefficients.from_data(
            data_dict[CircleType.ASJ][1],
            data_dict[CircleType.ED][1],
            metrics_ed_vol.q75_diff
        )
        
        # Update UI
        self.labelEcondMIP.setText(f"{coeff_mip.econd:.2f}")
        self.labelEpresMIP.setText(f"{coeff_mip.epres:.2f}")
        self.labelEcondVolume.setText(f"{coeff_vol.econd:.2f}")
        self.labelEpresVolume.setText(f"{coeff_vol.epres:.2f}")

    def _update_histogram(self):
        """Update histogram display."""
        data_asj_mip, data_asj_vol = self._get_circle_data_cached(CircleType.ASJ)
        data_ed_mip, data_ed_vol = self._get_circle_data_cached(CircleType.ED)
            
        if data_asj_mip is None or data_asj_vol is None:
                return
            
        # Update MIP histogram
        self.histogram_canvas_mip.update_data(data_asj_mip, data_ed_mip)
            
        # Update Volume histogram
        self.histogram_canvas_volume.update_data(data_asj_vol, data_ed_vol)

    def open_folder(self):
        """Open folder and show file tree."""
        folder = QFileDialog.getExistingDirectory(self, 'Open Folder')
        
        if not folder:
            return
        
        self._current_folder = Path(folder)
        self._update_file_list()
        
        # Show file tree
        self.file_model.setRootPath(str(self._current_folder))
        self.file_tree.setRootIndex(self.file_model.index(str(self._current_folder)))
        self.file_dock.show()
        
        # Load first file if available
        if self._file_list:
            self._current_file_index = 0
            self._load_file(self._file_list[0])
    
    def _on_file_tree_clicked(self, index):
        """Handle file tree click."""
        file_path = Path(self.file_model.filePath(index))
        
        if file_path.is_file() and file_path.suffix == '.dat':
            self._load_file(file_path)
    
    def _update_file_list(self):
        """Update list of .dat files in current folder."""
        if self._current_folder is None:
            self._file_list = []
            return
        
        self._file_list = sorted(self._current_folder.glob('*.dat'))
    
    def load_previous_file(self):
        """Load previous file in folder."""
        if not self._file_list or self._current_file_index <= 0:
            return
        
        self._current_file_index -= 1
        self._load_file(self._file_list[self._current_file_index])
    
    def load_next_file(self):
        """Load next file in folder."""
        if not self._file_list or self._current_file_index >= len(self._file_list) - 1:
            return
        
        self._current_file_index += 1
        self._load_file(self._file_list[self._current_file_index])
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Viewer()
    window.show()
    sys.exit(app.exec_())
