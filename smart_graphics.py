"""
Smart graphics view with zoom support.
"""
from PySide2.QtCore import QPointF, Qt, Signal
from PySide2.QtGui import QWheelEvent
from PySide2.QtWidgets import QGraphicsView


class SmartGraphicsView(QGraphicsView):
    zoomChanged = Signal(float, QPointF)
    panChanged = Signal(QPointF)
    centerCirclesRequested = Signal()
    scaleChanged = Signal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._base_scale = 1.0
        self._syncing = False
        self._panning = False

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.0015 ** event.angleDelta().y()
            factor = max(0.5, min(factor, 2.0))
            
            # Get current scale in comparison  with base
            current_scale = self.transform().m11() / self._base_scale
            new_scale = current_scale * factor
            
            # Deny zoom out less the base scale
            if new_scale < 1:
                return
                        
            self.scale(factor, factor)
            
            # Emit signals
            new_center = self.mapToScene(self.viewport().rect().center())
            self.zoomChanged.emit(factor, new_center)
            self.scaleChanged.emit(self.transform().m11())

        else:
            event.ignore()
    
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.centerCirclesRequested.emit()
        else:
            super().mouseDoubleClickEvent(event)
            
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._panning = True
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self._panning and not self._syncing:
            center = self.mapToScene(self.viewport().rect().center())
            self.panChanged.emit(center)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._panning = False
        super().mouseReleaseEvent(event)
            
    def sync_zoom(self, factor: float, center: QPointF):
        """Sync zoom with other view."""
        self._syncing = True
        
        current_scale = self.transform().m11() / self._base_scale
        new_scale = current_scale * factor
        
        if new_scale >= 0.995:
            self.scale(factor, factor)
            self.centerOn(center)
            self.scaleChanged.emit(self.transform().m11())
        
        self._syncing = False
        
    def sync_pan(self, center: QPointF):
        """Sync panning with other view."""
        if not self._panning:
            self._syncing = True
            self.centerOn(center)
            self._syncing = False
    
    def fitInView(self, rect, mode=Qt.IgnoreAspectRatio):
        """Override to remember base scale."""
        super().fitInView(rect, mode)
        self._base_scale = self.transform().m11()
    
    def showEvent(self, event):
        """Fit scene on first show."""
        super().showEvent(event)
        if self.scene():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
