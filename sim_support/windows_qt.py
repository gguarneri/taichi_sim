# =======================
# Importacao de pacotes de uso geral
# =======================
import numpy as np
from PyQt6.QtWidgets import *
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageWidget

# -----------------------------------------------
# Codigo para visualizacao das janelas de simulacao
# -----------------------------------------------
# Image View class
class ImageView(pg.ImageView):
    # constructor which inherit original
    # ImageView
    def __init__(self, *args, **kwargs):
        pg.ImageView.__init__(self, *args, **kwargs)


# Window class
class Window(QMainWindow):
    def __init__(self, title=None, geometry=None, nsteps=100, dx=1.0, dy=1.0, dt=1.0):
        super().__init__()

        # setting geometry
        if geometry is None:
            nx = 300
            ny = 300
            self.setGeometry(200, 50, nx, ny)
        else:
            nx = geometry[2]
            ny = geometry[3]
            self.setGeometry(*geometry)
            
        # setting title
        if title is None:
            self.setWindowTitle(f"{nx}x{ny} Grid x {nsteps} iterations - dx = {dx} m x dy = {dy} m x dt = {dt} s")
        else:
            self.setWindowTitle(title)

        # setting animation
        self.isAnimated()

        # setting image
        self.image = np.random.normal(size=(geometry[2], geometry[3]))

        # showing all the widgets
        self.show()

        # creating a widget object
        self.widget = QWidget()

        # setting configuration options
        pg.setConfigOptions(antialias=True)

        # creating image view view object
        self.imv = RawImageWidget()

        # setting image to image view
        self.imv.setImage(self.image, levels=[-0.1, 0.1])

        # Creating a grid layout
        self.layout = QGridLayout()

        # setting this layout to the widget
        self.widget.setLayout(self.layout)

        # plot window goes on right side, spanning 3 rows
        self.layout.addWidget(self.imv, 0, 0, 4, 1)

        # setting this widget as central widget of the main window
        self.setCentralWidget(self.widget)
