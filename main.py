"""
OCT image viewer with MVC architecture.

asgilliard
"""
import logging
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import darkdetect
import numpy as np
import pyqtgraph as pg
from matplotlib import colormaps
from PySide2.QtCore import QPointF, Qt, QTimer
from PySide2.QtGui import QBrush, QColor, QImage, QPen, QPixmap
from PySide2.QtWidgets import (
    QAction,
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

# Constants
IMAGE_SIZE = 512
DEFAULT_Z = 256
DEFAULT_MIP_LAYERS = 25
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
            q75_diff=float(np.percentile(data, 75)),
            variance=float(np.std(data))**2,
            size=int(data.size)
        )


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
        
    def load(self, file_path: Path) -> bool:
        """Load image data from file."""
        self.clear()
        
        try:
            if self._use_memmap:
                # Memory-mapped mode: lazy loading from disk
                self._memmap = np.memmap(
                    str(file_path), 
                    dtype=np.uint8, 
                    mode='r', 
                    shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)
                )
                self.data = self._memmap
                logger.info(f"Loaded as memmap: {file_path}")
            else:
                # RAM mode: load entire file into memory (default, faster)
                with open(file_path, 'rb') as f:
                    self.data = np.frombuffer(
                        f.read(),
                        dtype=np.uint8
                    ).reshape(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)
                    
                logger.info(f"Loaded into RAM: {file_path}")
            
            # Clear MIP cache on new file load
            if self._cache_mip:
                self._mip_cache.clear()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            return False
    
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
        z_max = min(IMAGE_SIZE, z + (layers - half_layers))
        return np.max(self.data[z_min:z_max, :, :], axis=0)
    
    def _is_valid_z(self, z: int) -> bool:
        """Check if z coordinate is valid."""
        return 0 <= z < IMAGE_SIZE
    
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
        if not (0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE):
            return np.array([])
        
        # Z range
        half1 = layers // 2
        half2 = layers - half1
        z_min = max(0, z - half1)
        z_max = min(IMAGE_SIZE, z + half2)
        
        # Get circle mask
        if radius not in self._mask_cache:
            self._mask_cache[radius] = self._create_circle_mask(radius)
        mask = self._mask_cache[radius]
        
        # Region borders XY
        x_min = max(0, x - radius)
        x_max = min(IMAGE_SIZE, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(IMAGE_SIZE, y + radius + 1)
        
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


class PyQtGraphHistogram(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        self.setBackground('w')
        self.curve1 = self.plot(pen=pg.mkPen('r', width=1), fillLevel=0, brush=(255,0,0,100))
        self.curve2 = self.plot(pen=pg.mkPen('g', width=1), fillLevel=0, brush=(0,255,0,100))
        
    def update_data(self, data1, data2):
        h1, edges = np.histogram(data1, bins=256, range=(0, 256))
        h2, _ = np.histogram(data2, bins=256, range=(0, 256))
        
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
            bounded_x = max(self.radius, min(scene_rect.width() - self.radius, value.x()))
            bounded_y = max(self.radius, min(scene_rect.height() - self.radius, value.y()))

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

    def create_circles(self, scene: QGraphicsScene, scene_mip: QGraphicsScene):
        """Create circle pairs for both scenes."""
        colors = {CircleType.ASJ: QColor(255, 0, 0), CircleType.ED: QColor(0, 255, 0)}

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

    def set_callbacks(self, sync_callback):
        """Set all callbacks."""
        self._sync_callback = sync_callback

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
        self._lut_cache = {}  # lookup table
        
    def _build_color_lut(self, cmap_name: str, invert: bool = False) -> np.ndarray:
        """Build color lookup table"""
        cache_key = (cmap_name, invert)
        if cache_key in self._lut_cache:
            return self._lut_cache[cache_key]
            
        if cmap_name not in self._cmap_cache:
            self._cmap_cache[cmap_name] = colormaps.get_cmap(cmap_name)
            
        cmap = self._cmap_cache[cmap_name]
        indices = np.linspace(0, 1, 256)
        if invert:
            indices = 1.0 - indices
            
        rgba = cmap(indices)  # (256, 4) float
        rgb = (rgba[:, :3] * 255).astype(np.uint8)
        
        self._lut_cache[cache_key] = rgb
        return rgb
        
    def array_to_qimage(self, arr: np.ndarray, invert: bool = False, cmap: str = 'gray') -> QImage:
        """Convert numpy array to QImage - reliable version"""
        if arr.size == 0:
            return QImage()
            
        height, width = arr.shape
        
        # Fast path for grayscale
        if cmap == 'gray':
            if invert:
                arr = 255 - arr
            bytes_per_line = width
            qimg = QImage(arr.data, width, height, bytes_per_line, QImage.Format_Grayscale8) # type: ignore
            return qimg.copy()
        
        # Color maps
        lut = self._build_color_lut(cmap, invert)
        rgb_data = lut[arr]
        
        # Ensure contiguous memory
        if not rgb_data.flags['C_CONTIGUOUS']:
            rgb_data = np.ascontiguousarray(rgb_data)
            
        bytes_per_line = 3 * width
        
        # Create QImage from buffer - this is the standard approach
        qimg = QImage(rgb_data.data, width, height, bytes_per_line, QImage.Format_RGB888) # type: ignore
        return qimg.copy()  # Copy is necessary to keep data alive
    
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
        self._setup_palette_menu()
        self._setup_scenes()
        self._setup_sliders()
        self._setup_theme_timer()
        self._setup_histogram()
        self._setup_metrics_table()
        self._setup_circles()
        self._setup_connections()

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
            # self.histogram_canvas._setup_style(new_theme)
            self.update_views(self.sliderZ.value(), update_histogram=False)

    def _setup_scenes(self):
        """Initialize graphics scenes."""
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, IMAGE_SIZE, IMAGE_SIZE)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.scene_mip = QGraphicsScene(self)
        self.scene_mip.setSceneRect(0, 0, IMAGE_SIZE, IMAGE_SIZE)
        self.graphicsView_MIP.setScene(self.scene_mip)
        self.graphicsView_MIP.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_MIP.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

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
        self.histogram_canvas = PyQtGraphHistogram()
        histogram_layout = QVBoxLayout(self.histogramWidget)
        histogram_layout.addWidget(self.histogram_canvas)

    def _setup_metrics_table(self):
        """Initialize metrics table."""
        self.metrics_table = MetricsTable(self.tableMetrics)

    def _setup_circles(self):
        """Initialize measurement circles."""
        self.circle_manager.create_circles(self.scene, self.scene_mip)
        self.circle_manager.set_callbacks(self._on_circles_changed)

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

    def open_file(self):
        """Open and load image file."""
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open .dat', '', 'DAT Files (*.dat)')
        if not file_name:
            return

        self._clear_all()

        if self.data_manager.load(Path(file_name)):
            self.update_views(DEFAULT_Z)
            self.pathLabel.setText(str(Path(file_name)))

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

    # def update_views(self, z: int, update_histogram: bool = True):
    #     """Update all views for given Z position."""

    #     # Render main slice
    #     slice_data = self.data_manager.get_slice(z)
    #     if slice_data is not None:
    #         pixmap = self.renderer.get_slice_pixmap(
    #             slice_data, z, self._current_theme, self._current_palette
    #         )
    #         self._update_scene_pixmap(self.scene, pixmap, is_mip=False)

    #     # Render MIP
    #     layers = self.sliderMIP.value()
    #     mip_data = self.data_manager.compute_mip(z, layers)
    #     if mip_data is not None:
    #         pixmap = self.renderer.get_mip_pixmap(
    #             mip_data, z, layers, self._current_theme, self._current_palette
    #         )
    #         self._update_scene_pixmap(self.scene_mip, pixmap, is_mip=True)

    #     # Fit views
    #     self.graphicsView.fitInView(0, 0, IMAGE_SIZE, IMAGE_SIZE, Qt.KeepAspectRatio)
    #     self.graphicsView_MIP.fitInView(0, 0, IMAGE_SIZE, IMAGE_SIZE, Qt.KeepAspectRatio)

    #     # Update metrics and histogram
    #     self._update_metrics()
    #     if update_histogram:
    #         self._update_histogram()
    
    def update_views(self, z: int, update_histogram: bool = True):
        """Update all views for given Z position."""
        t0 = time.perf_counter()
        
        # Render main slice
        slice_data = self.data_manager.get_slice(z)
        t1 = time.perf_counter()
        if slice_data is not None:
            pixmap = self.renderer.get_slice_pixmap(
                slice_data, z, self._current_theme, self._current_palette
            )
            self._update_scene_pixmap(self.scene, pixmap, is_mip=False)
        t2 = time.perf_counter()
        
        # Render MIP
        layers = self.sliderMIP.value()
        mip_data = self.data_manager.compute_mip(z, layers)
        t3 = time.perf_counter()
        if mip_data is not None:
            pixmap = self.renderer.get_mip_pixmap(
                mip_data, z, layers, self._current_theme, self._current_palette
            )
            self._update_scene_pixmap(self.scene_mip, pixmap, is_mip=True)
        t4 = time.perf_counter()
        
        # Fit views
        self.graphicsView.fitInView(0, 0, IMAGE_SIZE, IMAGE_SIZE, Qt.KeepAspectRatio)
        self.graphicsView_MIP.fitInView(0, 0, IMAGE_SIZE, IMAGE_SIZE, Qt.KeepAspectRatio)
        t5 = time.perf_counter()
        
        # Update metrics and histogram
        self._update_metrics()
        t6 = time.perf_counter()
        if update_histogram:
            self._update_histogram()
        t7 = time.perf_counter()
        
        print(f"get_slice: {(t1-t0)*1000:.1f}ms, "
              f"render_slice: {(t2-t1)*1000:.1f}ms, "
              f"compute_mip: {(t3-t2)*1000:.1f}ms, "
              f"render_mip: {(t4-t3)*1000:.1f}ms, "
              f"fitInView: {(t5-t4)*1000:.1f}ms, "
              f"metrics: {(t6-t5)*1000:.1f}ms, "
              f"histogram: {(t7-t6)*1000:.1f}ms")

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
        """Update metrics table."""
        if self.data_manager.data is None:
            return
        
        data_dict = {}
        
        for circle_type in CircleType:
            data_mip, data_vol = self._get_circle_data_cached(circle_type)
            if data_mip is None:
                return
            data_dict[circle_type] = (data_mip, data_vol)

        self.metrics_table.update(
            self.circle_manager.get_position(CircleType.ASJ),
            self.circle_manager.get_position(CircleType.ED),
            Metrics.from_data(data_dict[CircleType.ASJ][0]),
            Metrics.from_data(data_dict[CircleType.ED][0]),
            Metrics.from_data(data_dict[CircleType.ASJ][1]),
            Metrics.from_data(data_dict[CircleType.ED][1])
        )

    def _update_histogram(self):
        """Update histogram display."""
        data_asj, _ = self._get_circle_data_cached(CircleType.ASJ)
        data_ed, _ = self._get_circle_data_cached(CircleType.ED)
            
        if data_asj is None or data_ed is None:
            return
                
        self.histogram_canvas.update_data(data_asj, data_ed)

    def resizeEvent(self, event):
        """Handle window resize."""
        if self.data_manager.data is not None:
            self.graphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self.graphicsView_MIP.fitInView(self.scene_mip.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Viewer()
    window.show()
    sys.exit(app.exec_())
