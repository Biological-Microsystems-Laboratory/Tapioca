import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image, ImageOps
import numpy as np

class ImageModifier(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.original_image = None
        self.modified_image = None

    def initUI(self):
        self.setWindowTitle('Image Modifier')
        
        # Create buttons
        self.open_button = QPushButton('Open Image', self)
        self.open_button.clicked.connect(self.open_image)
        
        self.modify_button = QPushButton('Modify Image', self)
        self.modify_button.clicked.connect(self.modify_image)
        self.modify_button.setEnabled(False)
        
        # Create labels for displaying images
        self.original_label = QLabel(self)
        self.modified_label = QLabel(self)
        
        # Create layouts
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.modify_button)
        
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.original_label)
        image_layout.addWidget(self.modified_label)
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addLayout(image_layout)
        
        self.setLayout(main_layout)
        
        self.show()

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.original_image = Image.open(file_path)
            self.display_image(self.original_image, self.original_label)
            self.modify_button.setEnabled(True)

    def modify_image(self):
        if self.original_image:
            # Example modification: convert to grayscale
            self.modified_image = ImageOps.grayscale(self.original_image)
            self.display_image(self.modified_image, self.modified_label)

    def display_image(self, image, label):
        image = image.copy()  # Create a copy to avoid modifying the original
        image.thumbnail((300, 300))  # Resize image to fit in the window
        qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageModifier()
    sys.exit(app.exec_())