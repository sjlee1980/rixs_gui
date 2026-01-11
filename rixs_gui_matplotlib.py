import sys
import os
import gzip
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# --- Logic derived from pkl2dict.py ---
try:
    import cPickle
except ImportError:
    import pickle as cPickle

# --- Universal Qt & Matplotlib Backend Setup ---
try:
    # Attempt PyQt6 Import
    from PyQt6 import QtWidgets, QtCore, QtGui
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QFileDialog, QListWidget, 
                                 QAbstractItemView, QLabel, QLineEdit, QTextEdit, QSplitter)
    from PyQt6.QtCore import Qt
    matplotlib.use('QtAgg') 
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    EXEC_FUNC = "exec"
except ImportError:
    # Fallback to PyQt5 Import
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QFileDialog, QListWidget, 
                                 QAbstractItemView, QLabel, QLineEdit, QTextEdit, QSplitter)
    from PyQt5.QtCore import Qt
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    EXEC_FUNC = "exec_"

class RIXSGui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RIXS Analyzer: Multi-Scan & Sample ID")
        self.setGeometry(100, 100, 1200, 800)
        self.raw_data = {}
        self.init_ui()

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Sidebar
        sidebar = QVBoxLayout()
        self.load_btn = QPushButton("Load .pkl.gz Files")
        self.load_btn.clicked.connect(self.load_files)
        sidebar.addWidget(self.load_btn)

        sidebar.addWidget(QLabel("Scans (Hold Ctrl to select multiple):"))
        self.scan_list = QListWidget()
        self.scan_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.scan_list.itemSelectionChanged.connect(self.update_plot)
        sidebar.addWidget(self.scan_list)

        # Plotting
        plot_container = QVBoxLayout()
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_container.addWidget(self.toolbar)
        plot_container.addWidget(self.canvas)

        main_layout.addLayout(sidebar, 1)
        main_layout.addLayout(plot_container, 4)

    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open RIXS Data")
        for f in files:
            try:
                # Use the pkl2dict logic 
                with gzip.GzipFile(f, "rb") as gz:
                    try:
                        data = cPickle.load(gz)
                    except UnicodeDecodeError:
                        data = cPickle.load(gz, encoding='latin1') [cite: 1]
                
                fname = os.path.basename(f)
                self.raw_data[fname] = data
                # Populate list with folder names from data [cite: 1, 60]
                for scan_name in data.keys():
                    if isinstance(data[scan_name], dict):
                        self.scan_list.addItem(f"{fname} | {scan_name}")
            except Exception as e:
                print(f"Error loading {f}: {e}")

    def update_plot(self):
        selected = self.scan_list.selectedItems()
        if not selected:
            return
        
        self.ax.clear()
        for item in selected:
            f_key, s_key = item.text().split(" | ")
            rixs = self.raw_data[f_key][s_key]
            
            # Use nominal_mono for X and bin_centers for Y
            x = rixs.get('valid_nominal_mono', rixs['nominal_mono'])
            y = rixs['bin_centers']
            
            # Plane 0 for non-corrected, Plane 1 for corrected
            z_key = 'rixs_plane1_norm' if 'valid_nominal_mono' in rixs else 'rixs_plane0_norm'
            z_key = 'rixs_plane0_norm'

            # Sum over scans as per example
            z = np.sum(rixs[z_key], axis=0) 
            
            self.ax.pcolor(x, y, z.T, shading='auto')
            self.ax.set_title(f"RIXS: {s_key}")
            
        self.canvas.draw()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = RIXSGui()
    window.show()
    getattr(app, EXEC_FUNC)()