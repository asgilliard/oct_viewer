import sys
import os
import numpy as np

from PySide6.QtWidgets import (QApplication, QMainWindow, QFileDialog, 
    QGraphicsScene, QGraphicsView)
from PySide6.QtGui import QPixmap, QImage, QAction
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import Qt

from design_ui import Ui_MainWindow
# from PySide6.QtGui import QFontDatabase


class Viewer(QMainWindow, Ui_MainWindow):
    graphicsView: QGraphicsView
    actionOpen: QAction
    
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # создаём сцену
        self.scene = QGraphicsScene(self)
        self.graphicsView.setScene(self.scene)  # graphicsView - имя QGraphicsView из .ui

        # меню "Файл → Открыть"
        self.actionOpen.triggered.connect(self.open_file)
        
    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Открыть .dat", "", "DAT Files (*.dat)")
        if not file_name:
            return
        
        # Загружаем dat как массив 512x512 uint8
        data = np.fromfile(file_name, dtype=np.uint8)
        if data.size != 512 * 512:
            print("Файл неверного разрешения.")
            return
        
        image = data.reshape((512, 512))
        
        # создаём QImage (grayscale)
        qimage = QImage(image.data, 512, 512, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        
        # загружаем в сцену
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.graphicsView.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = Viewer()
    window.show()
 
    sys.exit(app.exec())
