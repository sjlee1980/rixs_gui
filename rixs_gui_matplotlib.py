import sys
import os
import gzip
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import RectangleSelector

# Suppress NumPy deprecation warnings from unpickling older data
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")

# --- Python 2/3 Pickle Compatibility ---
try:
    import cPickle
except ImportError:
    import pickle as cPickle

# --- Universal Qt & Matplotlib Backend Setup ---
try:
    from PyQt6 import QtWidgets, QtCore, QtGui
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QGridLayout, QPushButton, QFileDialog, QTreeWidget, QTreeWidgetItem, 
                                 QAbstractItemView, QLabel, QLineEdit, QSplitter, QComboBox, QSlider)
    from PyQt6.QtCore import Qt
    matplotlib.use('QtAgg') 
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    EXEC_FUNC = "exec"
    USER_ROLE = Qt.ItemDataRole.UserRole
    EXT_SEL = QAbstractItemView.SelectionMode.ExtendedSelection
except ImportError:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QGridLayout, QPushButton, QFileDialog, QTreeWidget, QTreeWidgetItem, 
                                 QAbstractItemView, QLabel, QLineEdit, QSplitter, QComboBox, QSlider)
    from PyQt5.QtCore import Qt
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    EXEC_FUNC = "exec_"
    USER_ROLE = Qt.UserRole
    EXT_SEL = QAbstractItemView.ExtendedSelection

class RIXSGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RIXS Analyzer: Fixed Axes & Smooth Dragging")
        self.setGeometry(50, 50, 1600, 1000)
        self.raw_data = {}
        
        self.current_data = None
        self.map_img = None
        self.selector = None 
        self.cax = None      
        
        # Line objects for smooth updating
        self.pfy_line = None
        self.xes_line = None
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ==================== SIDEBAR ====================
        sidebar = QVBoxLayout()
        
        self.load_btn = QPushButton("Load .pkl.gz Files")
        self.load_btn.clicked.connect(self.load_files)
        sidebar.addWidget(self.load_btn)

        sidebar.addWidget(QLabel("Data Hierarchy (Ctrl+Click to sum):"))
        self.scan_tree = QTreeWidget()
        self.scan_tree.setHeaderLabel("Files / Scans")
        self.scan_tree.setSelectionMode(EXT_SEL)
        self.scan_tree.itemSelectionChanged.connect(self.on_tree_selection) 
        sidebar.addWidget(self.scan_tree)

        color_group = QtWidgets.QGroupBox("Color Settings")
        color_layout = QGridLayout()
        
        color_layout.addWidget(QLabel("CMap:"), 0, 0)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'gray', 'seismic', 'jet'])
        self.cmap_combo.currentTextChanged.connect(self.update_color_map)
        color_layout.addWidget(self.cmap_combo, 0, 1)

        color_layout.addWidget(QLabel("Min:"), 1, 0)
        self.slider_min = QSlider(Qt.Orientation.Horizontal)
        self.slider_min.setRange(0, 100)
        self.slider_min.setValue(0)
        self.slider_min.valueChanged.connect(self.update_color_scale)
        color_layout.addWidget(self.slider_min, 1, 1)

        color_layout.addWidget(QLabel("Max:"), 2, 0)
        self.slider_max = QSlider(Qt.Orientation.Horizontal)
        self.slider_max.setRange(0, 100)
        self.slider_max.setValue(100)
        self.slider_max.valueChanged.connect(self.update_color_scale)
        color_layout.addWidget(self.slider_max, 2, 1)
        
        color_group.setLayout(color_layout)
        sidebar.addWidget(color_group)

        # ROI PFY
        roi_pfy_group = QtWidgets.QGroupBox("PFY ROI (Emission Y-Axis)")
        roi_pfy_layout = QGridLayout()
        self.roi_pfy_start = QLineEdit()
        self.roi_pfy_start.editingFinished.connect(self.update_integrations_from_text)
        roi_pfy_layout.addWidget(QLabel("Start:"), 0, 0)
        roi_pfy_layout.addWidget(self.roi_pfy_start, 0, 1)
        self.roi_pfy_end = QLineEdit()
        self.roi_pfy_end.editingFinished.connect(self.update_integrations_from_text)
        roi_pfy_layout.addWidget(QLabel("End:"), 1, 0)
        roi_pfy_layout.addWidget(self.roi_pfy_end, 1, 1)
        roi_pfy_group.setLayout(roi_pfy_layout)
        sidebar.addWidget(roi_pfy_group)

        # ROI XES
        roi_xes_group = QtWidgets.QGroupBox("XES ROI (Excitation X-Axis)")
        roi_xes_layout = QGridLayout()
        self.roi_xes_start = QLineEdit()
        self.roi_xes_start.editingFinished.connect(self.update_integrations_from_text)
        roi_xes_layout.addWidget(QLabel("Start:"), 0, 0)
        roi_xes_layout.addWidget(self.roi_xes_start, 0, 1)
        self.roi_xes_end = QLineEdit()
        self.roi_xes_end.editingFinished.connect(self.update_integrations_from_text)
        roi_xes_layout.addWidget(QLabel("End:"), 1, 0)
        roi_xes_layout.addWidget(self.roi_xes_end, 1, 1)
        roi_xes_group.setLayout(roi_xes_layout)
        sidebar.addWidget(roi_xes_group)

        # ==================== PLOTTING AREA ====================
        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.gs = gridspec.GridSpec(2, 3, width_ratios=[4, 1.2, 0.2], height_ratios=[1, 4], 
                                    wspace=0.1, hspace=0.1)

        self.ax_pfy = self.figure.add_subplot(self.gs[0, 0])
        self.ax_map = self.figure.add_subplot(self.gs[1, 0], sharex=self.ax_pfy)
        self.ax_xes = self.figure.add_subplot(self.gs[1, 1], sharey=self.ax_map)
        self.cax = self.figure.add_subplot(self.gs[1, 2])
        
        plt.setp(self.ax_pfy.get_xticklabels(), visible=False)
        plt.setp(self.ax_xes.get_yticklabels(), visible=False)

        plot_container = QVBoxLayout()
        plot_container.addWidget(self.toolbar)
        plot_container.addWidget(self.canvas)

        main_layout.addLayout(sidebar, 1)
        main_layout.addLayout(plot_container, 4)

    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open RIXS Data")
        if not files: return
        
        for f in files:
            try:
                with gzip.GzipFile(f, "rb") as gz:
                    try: data = cPickle.load(gz)
                    except UnicodeDecodeError: data = cPickle.load(gz, encoding='latin1')
                
                fname = os.path.basename(f)
                self.raw_data[fname] = data
                
                file_item = QTreeWidgetItem(self.scan_tree)
                file_item.setText(0, fname)
                file_item.setExpanded(True)
                
                for scan_name in data.keys():
                    if isinstance(data[scan_name], dict):
                        scan_item = QTreeWidgetItem(file_item)
                        scan_item.setText(0, str(scan_name))
                        scan_item.setData(0, USER_ROLE, (fname, scan_name))
            except Exception as e:
                print(f"Error loading {f}: {e}")

    def get_summed_data(self):
        selected_items = self.scan_tree.selectedItems()
        if not selected_items: return None

        summed_z, common_x, common_y = None, None, None

        for item in selected_items:
            user_data = item.data(0, USER_ROLE)
            if not user_data: continue
            
            fname, sname = user_data
            if fname not in self.raw_data: continue
            rixs = self.raw_data[fname][sname]

            z_key = 'rixs_plane1_norm' if 'rixs_plane1_norm' in rixs else 'rixs_plane0_norm'
            x_key = 'valid_nominal_mono' if 'valid_nominal_mono' in rixs else 'nominal_mono'
            
            z = np.sum(rixs[z_key], axis=0) 
            x = rixs[x_key]
            y = rixs['bin_centers']

            if summed_z is None:
                summed_z = z
                common_x = x
                common_y = y
            else:
                if z.shape == summed_z.shape:
                    summed_z += z

        if summed_z is None: return None
        return {'x': common_x, 'y': common_y, 'z': summed_z}

    def on_tree_selection(self):
        self.current_data = self.get_summed_data()
        if not self.current_data: return
        d = self.current_data
        
        x_min, x_max = np.min(d['x']), np.max(d['x'])
        y_min, y_max = np.min(d['y']), np.max(d['y'])

        # Setup texts if empty
        if not self.roi_pfy_start.text(): self.roi_pfy_start.setText(f"{y_min:.2f}")
        if not self.roi_pfy_end.text(): self.roi_pfy_end.setText(f"{y_max:.2f}")
        if not self.roi_xes_start.text(): self.roi_xes_start.setText(f"{x_min:.2f}")
        if not self.roi_xes_end.text(): self.roi_xes_end.setText(f"{x_max:.2f}")

        try:
            y_s = float(self.roi_pfy_start.text())
            y_e = float(self.roi_pfy_end.text())
            x_s = float(self.roi_xes_start.text())
            x_e = float(self.roi_xes_end.text())
        except ValueError:
            y_s, y_e = y_min, y_max
            x_s, x_e = x_min, x_max

        # 1. Clear ALL plots fully only on new selection
        self.ax_map.clear()
        self.ax_pfy.clear()
        self.ax_xes.clear()
        self.pfy_line = None
        self.xes_line = None

        # 2. Draw map
        self.map_img = self.ax_map.pcolormesh(d['x'], d['y'], d['z'].T, shading='auto', 
                                              cmap=self.cmap_combo.currentText())
        
        # Hard-clamp the map limits immediately to prevent (0,0) stretching
        self.ax_map.set_xlim(x_min, x_max)
        self.ax_map.set_ylim(y_min, y_max)
        
        # 3. Setup Interactive ROI Selector
        if self.selector: self.selector.set_active(False)
        self.selector = RectangleSelector(
            self.ax_map, self.on_select_roi,
            useblit=True, button=[1], minspanx=5, minspany=5, interactive=True
        )
        self.selector.extents = (x_s, x_e, y_s, y_e)
        
        # 4. Colorbar
        self.cax.clear()
        self.figure.colorbar(self.map_img, cax=self.cax)
        self.update_color_scale()
        
        # 5. Build integration lines
        self.update_integrations()

    def on_select_roi(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        self.roi_xes_start.setText(f"{x_min:.2f}")
        self.roi_xes_end.setText(f"{x_max:.2f}")
        self.roi_pfy_start.setText(f"{y_min:.2f}")
        self.roi_pfy_end.setText(f"{y_max:.2f}")
        
        self.update_integrations()

    def update_integrations_from_text(self):
        self.update_integrations()

    def update_integrations(self):
        if not self.current_data: return
        d = self.current_data
        
        try:
            y_s = float(self.roi_pfy_start.text())
            y_e = float(self.roi_pfy_end.text())
            x_s = float(self.roi_xes_start.text())
            x_e = float(self.roi_xes_end.text())
        except ValueError:
            return

        x_min, x_max = np.min(d['x']), np.max(d['x'])
        y_min, y_max = np.min(d['y']), np.max(d['y'])

        # Clamp text fields strictly to data bounds
        x_s, x_e = max(x_min, x_s), min(x_max, x_e)
        y_s, y_e = max(y_min, y_s), min(y_max, y_e)
        
        self.roi_xes_start.setText(f"{x_s:.2f}")
        self.roi_xes_end.setText(f"{x_e:.2f}")
        self.roi_pfy_start.setText(f"{y_s:.2f}")
        self.roi_pfy_end.setText(f"{y_e:.2f}")

        if self.selector:
            self.selector.extents = (x_s, x_e, y_s, y_e)

        # ---------------- PFY ----------------
        idx_y_i = np.argmin(np.abs(d['y'] - y_s))
        idx_y_f = np.argmin(np.abs(d['y'] - y_e))
        pfy_curve = np.sum(d['z'][:, min(idx_y_i, idx_y_f):max(idx_y_i, idx_y_f)], axis=1)
        
        if self.pfy_line is None or self.pfy_line not in self.ax_pfy.lines:
            self.pfy_line, = self.ax_pfy.plot(d['x'], pfy_curve, color='lime')
            self.ax_pfy.set_ylabel("PFY")
            plt.setp(self.ax_pfy.get_xticklabels(), visible=False)
        else:
            self.pfy_line.set_ydata(pfy_curve)
            self.pfy_line.set_xdata(d['x'])
            self.ax_pfy.relim()
            self.ax_pfy.autoscale_view(scalex=False, scaley=True)

        # FIX: Safe removal of patches for newer Matplotlib ArtistList
        for patch in list(self.ax_pfy.patches): patch.remove()
        self.ax_pfy.axvspan(x_s, x_e, color='cyan', alpha=0.1) 

        # ---------------- XES ----------------
        idx_x_i = np.argmin(np.abs(d['x'] - x_s))
        idx_x_f = np.argmin(np.abs(d['x'] - x_e))
        xes_curve = np.sum(d['z'][min(idx_x_i, idx_x_f):max(idx_x_i, idx_x_f), :], axis=0)
        
        if self.xes_line is None or self.xes_line not in self.ax_xes.lines:
            self.xes_line, = self.ax_xes.plot(xes_curve, d['y'], color='cyan')
            self.ax_xes.set_xlabel("XES")
            plt.setp(self.ax_xes.get_yticklabels(), visible=False)
        else:
            self.xes_line.set_xdata(xes_curve)
            self.xes_line.set_ydata(d['y'])
            self.ax_xes.relim()
            self.ax_xes.autoscale_view(scalex=True, scaley=False)

        # FIX: Safe removal of patches for newer Matplotlib ArtistList
        for patch in list(self.ax_xes.patches): patch.remove()
        self.ax_xes.axhspan(y_s, y_e, color='lime', alpha=0.1)

        # ---------------- MAP LINES ----------------
        # FIX: Safe removal of lines for newer Matplotlib ArtistList
        for line in list(self.ax_map.lines): line.remove()
        self.ax_map.axhline(y_s, color='lime', linestyle='--', linewidth=1)
        self.ax_map.axhline(y_e, color='lime', linestyle='--', linewidth=1)
        self.ax_map.axvline(x_s, color='cyan', linestyle='--', linewidth=1)
        self.ax_map.axvline(x_e, color='cyan', linestyle='--', linewidth=1)

        self.canvas.draw()

    def update_color_scale(self):
        if self.map_img is None: return
        raw_data = self.map_img.get_array()
        d_min, d_max = np.min(raw_data), np.max(raw_data)
        s_min = self.slider_min.value() / 100.0
        s_max = self.slider_max.value() / 100.0
        new_vmin = d_min + (d_max - d_min) * s_min
        new_vmax = d_min + (d_max - d_min) * s_max
        if new_vmin >= new_vmax: new_vmax = new_vmin + 1e-6
        self.map_img.set_clim(new_vmin, new_vmax)
        self.canvas.draw()

    def update_color_map(self, cmap_name):
        if self.map_img:
            self.map_img.set_cmap(cmap_name)
            self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RIXSGui()
    window.show()
    getattr(app, EXEC_FUNC)()