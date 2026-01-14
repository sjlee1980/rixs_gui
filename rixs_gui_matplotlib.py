import sys
import os
import gzip
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar

# --- Logic derived from pkl2dict.py ---
try:
    import cPickle
except ImportError:
    import pickle as cPickle

# --- Universal Qt & Matplotlib Backend Setup ---
try:
    from PyQt6 import QtWidgets, QtCore, QtGui
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QGridLayout, QPushButton, QFileDialog, QTreeWidget, QTreeWidgetItem, 
                                 QAbstractItemView, QLabel, QLineEdit, QTextEdit, QSplitter, QComboBox, QSlider, QCheckBox)
    from PyQt6.QtCore import Qt
    matplotlib.use('QtAgg') 
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    EXEC_FUNC = "exec"
except ImportError:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QGridLayout, QPushButton, QFileDialog, QTreeWidget, QTreeWidgetItem, 
                                 QAbstractItemView, QLabel, QLineEdit, QTextEdit, QSplitter, QComboBox, QSlider, QCheckBox)
    from PyQt5.QtCore import Qt
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    EXEC_FUNC = "exec_"

class RIXSGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RIXS Analyzer: Stable Colorbar & Auto-Update")
        self.setGeometry(50, 50, 1600, 1000)
        self.raw_data = {}
        
        # Plotting objects placeholders
        self.map_img = None
        self.cbar = None
        self.roi_h_lines = []
        self.roi_v_lines = []
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ==================== SIDEBAR ====================
        sidebar = QVBoxLayout()
        
        # 1. File Loading
        self.load_btn = QPushButton("Load .pkl.gz Files")
        self.load_btn.clicked.connect(self.load_files)
        sidebar.addWidget(self.load_btn)

        # 2. Tree View
        sidebar.addWidget(QLabel("Data Hierarchy (Ctrl+Click to sum):"))
        self.scan_tree = QTreeWidget()
        self.scan_tree.setHeaderLabel("Files / Scans")
        self.scan_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.scan_tree.itemSelectionChanged.connect(self.update_plots) # Auto-update on selection
        sidebar.addWidget(self.scan_tree)

        # 3. Color Map Controls
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

        # 4. ROI Controls (PFY - Emission)
        roi_pfy_group = QtWidgets.QGroupBox("PFY ROI (Emission Y-Axis)")
        roi_pfy_layout = QGridLayout()
        
        roi_pfy_layout.addWidget(QLabel("Start:"), 0, 0)
        self.roi_pfy_start = QLineEdit("0")
        self.roi_pfy_start.editingFinished.connect(self.update_plots) # Updates on Enter or Click away
        roi_pfy_layout.addWidget(self.roi_pfy_start, 0, 1)
        
        roi_pfy_layout.addWidget(QLabel("End:"), 1, 0)
        self.roi_pfy_end = QLineEdit("1000")
        self.roi_pfy_end.editingFinished.connect(self.update_plots)
        roi_pfy_layout.addWidget(self.roi_pfy_end, 1, 1)
        
        roi_pfy_group.setLayout(roi_pfy_layout)
        sidebar.addWidget(roi_pfy_group)

        # 5. ROI Controls (XES - Excitation)
        roi_xes_group = QtWidgets.QGroupBox("XES ROI (Excitation X-Axis)")
        roi_xes_layout = QGridLayout()
        
        roi_xes_layout.addWidget(QLabel("Start:"), 0, 0)
        self.roi_xes_start = QLineEdit("0")
        self.roi_xes_start.editingFinished.connect(self.update_plots)
        roi_xes_layout.addWidget(self.roi_xes_start, 0, 1)
        
        roi_xes_layout.addWidget(QLabel("End:"), 1, 0)
        self.roi_xes_end = QLineEdit("2000")
        self.roi_xes_end.editingFinished.connect(self.update_plots)
        roi_xes_layout.addWidget(self.roi_xes_end, 1, 1)
        
        roi_xes_group.setLayout(roi_xes_layout)
        sidebar.addWidget(roi_xes_group)

        # ==================== PLOTTING AREA ====================
        # Single Figure with GridSpec
        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # GridSpec Layout: 
        # Top row: PFY (long)
        # Bottom row: Map (left) + XES (right) + Colorbar (far right)
        # width_ratios: [Map, XES, Colorbar]
        self.gs = gridspec.GridSpec(2, 3, width_ratios=[4, 1.2, 0.2], height_ratios=[1, 4], 
                                    wspace=0.1, hspace=0.1)

        # Create Axes
        self.ax_pfy = self.figure.add_subplot(self.gs[0, 0])
        self.ax_map = self.figure.add_subplot(self.gs[1, 0], sharex=self.ax_pfy)
        self.ax_xes = self.figure.add_subplot(self.gs[1, 1], sharey=self.ax_map)
        self.cax = self.figure.add_subplot(self.gs[1, 2]) # Permanent Colorbar Axis
        
        # Turn off redundant tick labels for shared axes
        plt.setp(self.ax_pfy.get_xticklabels(), visible=False)
        plt.setp(self.ax_xes.get_yticklabels(), visible=False)

        plot_container = QVBoxLayout()
        plot_container.addWidget(self.toolbar)
        plot_container.addWidget(self.canvas)

        # ==================== ASSEMBLY ====================
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
                        # FIXED: Convert integer scan_name to string to prevent crash
                        scan_item.setText(0, str(scan_name))
                        scan_item.setData(0, Qt.ItemDataRole.UserRole, (fname, scan_name))
            except Exception as e:
                print(f"Error loading {f}: {e}")

    def get_summed_data(self):
        """Sum selected maps and return composite data."""
        selected_items = self.scan_tree.selectedItems()
        if not selected_items: return None

        summed_z = None
        common_x = None
        common_y = None

        for item in selected_items:
            user_data = item.data(0, Qt.ItemDataRole.UserRole)
            if not user_data: continue
            
            fname, sname = user_data
            if fname not in self.raw_data: continue
            rixs = self.raw_data[fname][sname]

            #
            z_key = 'rixs_plane1_norm' if 'rixs_plane1_norm' in rixs else 'rixs_plane0_norm'
            x_key = 'valid_nominal_mono' if 'valid_nominal_mono' in rixs else 'nominal_mono'
            
            z = np.sum(rixs[z_key], axis=0) # Sum over scans
            x = rixs[x_key]
            y = rixs['bin_centers']

            if summed_z is None:
                summed_z = z
                common_x = x
                common_y = y
            else:
                # Simple check for shape compatibility
                if z.shape == summed_z.shape:
                    summed_z += z

        if summed_z is None: return None
        return {'x': common_x, 'y': common_y, 'z': summed_z}

    def update_plots(self):
        data = self.get_summed_data()
        if not data: return

        # 1. Validate and Clamp ROIs
        try:
            # PFY ROI (Emission / Y-Axis)
            y_min, y_max = data['y'][0], data['y'][-1]
            if y_min > y_max: y_min, y_max = y_max, y_min
            
            try: roi_y_start = float(self.roi_pfy_start.text())
            except: roi_y_start = 0.0
            
            try: roi_y_end = float(self.roi_pfy_end.text())
            except: roi_y_end = 1000.0
            
            # Initial setup or "Full Range" check
            if roi_y_start == 0 and roi_y_end == 1000:
                roi_y_start, roi_y_end = y_min, y_max
                self.roi_pfy_start.setText(f"{y_min:.2f}")
                self.roi_pfy_end.setText(f"{y_max:.2f}")
            
            # Clamp Logic
            if roi_y_end > y_max: 
                roi_y_end = y_max
                self.roi_pfy_end.setText(f"{y_max:.2f}")
            if roi_y_start < y_min:
                roi_y_start = y_min
                self.roi_pfy_start.setText(f"{y_min:.2f}")

            # XES ROI (Excitation / X-Axis)
            x_min, x_max = data['x'][0], data['x'][-1]
            try: roi_x_start = float(self.roi_xes_start.text())
            except: roi_x_start = 0.0
            
            try: roi_x_end = float(self.roi_xes_end.text())
            except: roi_x_end = 2000.0

            if roi_x_start == 0 and roi_x_end == 2000:
                roi_x_start, roi_x_end = x_min, x_max
                self.roi_xes_start.setText(f"{x_min:.2f}")
                self.roi_xes_end.setText(f"{x_max:.2f}")

        except ValueError:
            return 

        # 2. Plot Map (Center)
        self.ax_map.clear()
        self.map_img = self.ax_map.pcolormesh(data['x'], data['y'], data['z'].T, shading='auto', 
                                              cmap=self.cmap_combo.currentText())
        
        # Add ROI Lines
        self.ax_map.axhline(roi_y_start, color='lime', linestyle='--', linewidth=1, alpha=0.7)
        self.ax_map.axhline(roi_y_end, color='lime', linestyle='--', linewidth=1, alpha=0.7)
        self.ax_map.axvline(roi_x_start, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
        self.ax_map.axvline(roi_x_end, color='cyan', linestyle='--', linewidth=1, alpha=0.7)

        # 3. Plot PFY (Top)
        self.ax_pfy.clear()
        idx_y_i = np.argmin(np.abs(data['y'] - roi_y_start))
        idx_y_f = np.argmin(np.abs(data['y'] - roi_y_end))
        pfy_curve = np.sum(data['z'][:, min(idx_y_i, idx_y_f):max(idx_y_i, idx_y_f)], axis=1)
        self.ax_pfy.plot(data['x'], pfy_curve, color='lime')
        self.ax_pfy.set_ylabel("PFY")
        plt.setp(self.ax_pfy.get_xticklabels(), visible=False)

        # 4. Plot XES (Right)
        self.ax_xes.clear()
        idx_x_i = np.argmin(np.abs(data['x'] - roi_x_start))
        idx_x_f = np.argmin(np.abs(data['x'] - roi_x_end))
        xes_curve = np.sum(data['z'][min(idx_x_i, idx_x_f):max(idx_x_i, idx_x_f), :], axis=0)
        self.ax_xes.plot(xes_curve, data['y'], color='cyan') 
        self.ax_xes.set_xlabel("XES")
        plt.setp(self.ax_xes.get_yticklabels(), visible=False)

        # 5. Stable Colorbar Update
        # Instead of creating/removing, we just update the mappable inside the permanent axis
        self.cax.clear()
        self.cbar = self.figure.colorbar(self.map_img, cax=self.cax, orientation='vertical')
        
        self.update_color_scale() # Apply contrast
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