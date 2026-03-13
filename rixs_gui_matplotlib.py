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

# Fix for NumPy 2.0 unpickling legacy NumPy 1.x data (e.g., .pkl.gz)
if np.__version__.startswith("2."):
    import numpy._core as _core
    sys.modules["numpy.core"] = _core
    sys.modules["numpy.core.numeric"] = _core.numeric
    sys.modules["numpy.core.multiarray"] = _core.multiarray

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
                                 QAbstractItemView, QLabel, QLineEdit, QSplitter, QComboBox, QSlider, QMessageBox, QCheckBox)
    from PyQt6.QtCore import Qt
    matplotlib.use('QtAgg') 
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    EXEC_FUNC = "exec"
    USER_ROLE = Qt.ItemDataRole.UserRole
    SEL_MODE = QAbstractItemView.SelectionMode.ExtendedSelection
except ImportError:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QGridLayout, QPushButton, QFileDialog, QTreeWidget, QTreeWidgetItem, 
                                 QAbstractItemView, QLabel, QLineEdit, QSplitter, QComboBox, QSlider, QMessageBox, QCheckBox)
    from PyQt5.QtCore import Qt
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    EXEC_FUNC = "exec_"
    USER_ROLE = Qt.UserRole
    SEL_MODE = QAbstractItemView.ExtendedSelection

class RIXSGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RIXS Analyzer: Stable Flux-Corrected Counts")
        self.setGeometry(50, 50, 1600, 1000)
        self.raw_data = {}
        
        self.current_data = None
        self.map_img = None
        self.selector = None 
        self.cax = None      
        
        self.pfy_line = None
        self.xes_line = None
        
        self.current_pfy_x = None
        self.current_pfy_y = None
        self.current_xes_x = None
        self.current_xes_y = None
        
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

        sidebar.addWidget(QLabel("Data Hierarchy:"))
        self.scan_tree = QTreeWidget()
        self.scan_tree.setHeaderLabel("Files / Folders / Scans")
        self.scan_tree.setSelectionMode(SEL_MODE)
        self.scan_tree.itemSelectionChanged.connect(self.on_tree_selection)
        self.scan_tree.itemChanged.connect(self.on_checkbox_trigger) 
        sidebar.addWidget(self.scan_tree)

        sidebar.addSpacing(10)

        # Checkbox for Flux-Corrected Counts
        self.count_toggle = QCheckBox("Show Flux-Corrected Counts")
        self.count_toggle.clicked.connect(self.on_tree_selection)
        sidebar.addWidget(self.count_toggle)
        
        self.save_map_btn = QPushButton("Save Summed RIXS Map (.pkl.gz)")
        self.save_map_btn.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold; padding: 5px;")
        self.save_map_btn.clicked.connect(self.export_2d_map)
        sidebar.addWidget(self.save_map_btn)

        self.export_btn = QPushButton("Save Integrated 1D Curves (.txt)")
        self.export_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 5px;")
        self.export_btn.clicked.connect(self.export_1d_data)
        sidebar.addWidget(self.export_btn)
        
        sidebar.addSpacing(10)

        # --- Color Settings ---
        color_group = QtWidgets.QGroupBox("Color Settings")
        color_layout = QGridLayout()
        color_layout.addWidget(QLabel("CMap:"), 0, 0)
        self.cmap_combo = QComboBox(); self.cmap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'gray', 'seismic', 'jet'])
        self.cmap_combo.currentTextChanged.connect(self.update_color_map)
        color_layout.addWidget(self.cmap_combo, 0, 1)
        color_layout.addWidget(QLabel("Min:"), 1, 0)
        self.slider_min = QSlider(Qt.Orientation.Horizontal); self.slider_min.setRange(0, 100); self.slider_min.setValue(0)
        self.slider_min.valueChanged.connect(self.update_color_scale); color_layout.addWidget(self.slider_min, 1, 1)
        color_layout.addWidget(QLabel("Max:"), 2, 0)
        self.slider_max = QSlider(Qt.Orientation.Horizontal); self.slider_max.setRange(0, 100); self.slider_max.setValue(100)
        self.slider_max.valueChanged.connect(self.update_color_scale); color_layout.addWidget(self.slider_max, 2, 1)
        color_group.setLayout(color_layout); sidebar.addWidget(color_group)

        # --- ROI ---
        roi_pfy_group = QtWidgets.QGroupBox("PFY ROI (Emission Y-Axis)"); roi_pfy_layout = QGridLayout()
        self.roi_pfy_start = QLineEdit(); self.roi_pfy_start.editingFinished.connect(self.update_integrations_from_text); roi_pfy_layout.addWidget(QLabel("Start:"), 0, 0); roi_pfy_layout.addWidget(self.roi_pfy_start, 0, 1)
        self.roi_pfy_end = QLineEdit(); self.roi_pfy_end.editingFinished.connect(self.update_integrations_from_text); roi_pfy_layout.addWidget(QLabel("End:"), 1, 0); roi_pfy_layout.addWidget(self.roi_pfy_end, 1, 1)
        roi_pfy_group.setLayout(roi_pfy_layout); sidebar.addWidget(roi_pfy_group)

        roi_xes_group = QtWidgets.QGroupBox("XES ROI (Excitation X-Axis)"); roi_xes_layout = QGridLayout()
        self.roi_xes_start = QLineEdit(); self.roi_xes_start.editingFinished.connect(self.update_integrations_from_text); roi_xes_layout.addWidget(QLabel("Start:"), 0, 0); roi_xes_layout.addWidget(self.roi_xes_start, 0, 1)
        self.roi_xes_end = QLineEdit(); self.roi_xes_end.editingFinished.connect(self.update_integrations_from_text); roi_xes_layout.addWidget(QLabel("End:"), 1, 0); roi_xes_layout.addWidget(self.roi_xes_end, 1, 1)
        roi_xes_group.setLayout(roi_xes_layout); sidebar.addWidget(roi_xes_group)

        # --- Plots ---
        self.figure = plt.figure(figsize=(10, 8)); self.canvas = FigureCanvas(self.figure); self.toolbar = NavigationToolbar(self.canvas, self)
        # Fix: Spacing to prevent axis label overlap
        self.gs = gridspec.GridSpec(2, 3, width_ratios=[4, 1.2, 0.2], height_ratios=[1, 4], wspace=0.35, hspace=0.35)
        self.ax_pfy = self.figure.add_subplot(self.gs[0, 0]); self.ax_map = self.figure.add_subplot(self.gs[1, 0], sharex=self.ax_pfy); self.ax_xes = self.figure.add_subplot(self.gs[1, 1], sharey=self.ax_map); self.cax = self.figure.add_subplot(self.gs[1, 2])
        plt.setp(self.ax_pfy.get_xticklabels(), visible=False); plt.setp(self.ax_xes.get_yticklabels(), visible=False)

        plot_container = QVBoxLayout(); plot_container.addWidget(self.toolbar); plot_container.addWidget(self.canvas)
        main_layout.addLayout(sidebar, 1); main_layout.addLayout(plot_container, 4)

    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open RIXS Data")
        if not files: return
        self.scan_tree.blockSignals(True)
        for f in files:
            try:
                with gzip.GzipFile(f, "rb") as gz:
                    try: data = cPickle.load(gz)
                    except UnicodeDecodeError: data = cPickle.load(gz, encoding='latin1')
                fname = os.path.basename(f); self.raw_data[fname] = data
                file_item = QTreeWidgetItem(self.scan_tree); file_item.setText(0, fname); file_item.setExpanded(True)
                for scan_key in data.keys():
                    if isinstance(scan_key, str) and isinstance(data[scan_key], dict):
                        sample_name = data[scan_key].get('sample_name', 'Unknown')
                        folder_item = QTreeWidgetItem(file_item); folder_item.setText(0, f"{scan_key}: {sample_name}")
                        folder_item.setData(0, USER_ROLE, (fname, scan_key))
                        scans_list = data[scan_key].get('scan_numbers', str(scan_key))
                        if isinstance(scans_list, str): scans_list = [scans_list]
                        for s_num in scans_list:
                            scan_item = QTreeWidgetItem(folder_item); scan_item.setText(0, f"Scan {s_num}")
                            scan_item.setCheckState(0, Qt.CheckState.Checked)
            except Exception as e: print(f"Error loading {f}: {e}")
        self.scan_tree.blockSignals(False)

    def on_checkbox_trigger(self, item, column):
        if item.parent() and item.parent().data(0, USER_ROLE): self.on_tree_selection()

    def get_summed_data(self):
        """Sums checked scans and handles scaling by retrieving I0 flux ratio."""
        selected_items = self.scan_tree.selectedItems()
        if not selected_items: return None
        summed_z_norm, common_x, common_y = None, None, None
        summed_z_flux, summed_pfy_flux_multiplier = None, None

        for folder_item in selected_items:
            user_data = folder_item.data(0, USER_ROLE)
            if not user_data: continue
            fname, sname = user_data; rixs = self.raw_data[fname][sname]

            # Calculation: Retrieve I0 flux by ratio of raw/normalized maps
            raw_map = np.sum(rixs['rixs_plane0'], axis=0)
            norm_map = np.sum(rixs['rixs_plane0_norm'], axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                i0_array = np.nan_to_num(np.nanmean(raw_map / norm_map, axis=1), nan=1.0)

            for i in range(folder_item.childCount()):
                scan_item = folder_item.child(i)
                if scan_item.checkState(0) == Qt.CheckState.Checked:
                    z_key = 'rixs_plane1_norm' if 'rixs_plane1_norm' in rixs else 'rixs_plane0_norm'
                    z_norm = np.sum(rixs[z_key], axis=0)
                    x, y = rixs['valid_nominal_mono' if 'valid_nominal_mono' in rixs else 'nominal_mono'], rixs['bin_centers']
                    
                    # Apply incident-energy-dependent I0 scaling to the map for XES
                    z_flux = z_norm * i0_array[:, np.newaxis]
                    # Capture the initial I0 scaling for PFY summing
                    pfy_flux_factor = i0_array[0]

                    if summed_z_norm is None:
                        summed_z_norm, common_x, common_y = z_norm.copy(), x, y
                        summed_z_flux = z_flux.copy()
                        summed_pfy_flux_multiplier = pfy_flux_factor
                    elif z_norm.shape == summed_z_norm.shape:
                        summed_z_norm += z_norm
                        summed_z_flux += z_flux
                        summed_pfy_flux_multiplier += pfy_flux_factor

        return {'x': common_x, 'y': common_y, 'z_norm': summed_z_norm, 
                'z_flux': summed_z_flux, 'i0_scale': summed_pfy_flux_multiplier} if summed_z_norm is not None else None

    def on_tree_selection(self):
        new_data = self.get_summed_data()
        if not new_data: self.ax_map.clear(); self.ax_pfy.clear(); self.ax_xes.clear(); self.canvas.draw(); return
        self.current_data = new_data; d = self.current_data; x_min, x_max = np.min(d['x']), np.max(d['x']); y_min, y_max = np.min(d['y']), np.max(d['y'])
        
        if not self.roi_pfy_start.text(): self.roi_pfy_start.setText(f"{y_min:.2f}"); self.roi_pfy_end.setText(f"{y_max:.2f}")
        if not self.roi_xes_start.text(): self.roi_xes_start.setText(f"{x_min:.2f}"); self.roi_xes_end.setText(f"{x_max:.2f}")

        try: y_s, y_e = float(self.roi_pfy_start.text()), float(self.roi_pfy_end.text()); x_s, x_e = float(self.roi_xes_start.text()), float(self.roi_xes_end.text())
        except ValueError: y_s, y_e, x_s, x_e = y_min, y_max, x_min, x_max

        self.ax_map.clear(); self.ax_pfy.clear(); self.ax_xes.clear(); self.pfy_line = None; self.xes_line = None
        
        # Use scaled map for display if toggle is on
        disp_z = d['z_flux'] if self.count_toggle.isChecked() else d['z_norm']
        self.map_img = self.ax_map.pcolormesh(d['x'], d['y'], disp_z.T, shading='auto', cmap=self.cmap_combo.currentText())
        
        self.ax_map.set_xlabel("Incident Energy (eV)"); self.ax_map.set_ylabel("Emission Energy (eV)")
        self.ax_map.set_xlim(x_min, x_max); self.ax_map.set_ylim(y_min, y_max)
        
        if self.selector: self.selector.set_active(False)
        self.selector = RectangleSelector(self.ax_map, self.on_select_roi, useblit=True, button=[1], minspanx=5, minspany=5, interactive=True); self.selector.extents = (x_s, x_e, y_s, y_e)
        self.cax.clear(); self.figure.colorbar(self.map_img, cax=self.cax); self.update_color_scale(); self.update_integrations()

    def on_select_roi(self, eclick, erelease):
        x1, x2 = sorted([eclick.xdata, erelease.xdata]); y1, y2 = sorted([eclick.ydata, erelease.ydata])
        self.roi_xes_start.setText(f"{x1:.2f}"); self.roi_xes_end.setText(f"{x2:.2f}"); self.roi_pfy_start.setText(f"{y1:.2f}"); self.roi_pfy_end.setText(f"{y2:.2f}")
        self.update_integrations()

    def update_integrations_from_text(self): self.update_integrations()

    def update_integrations(self):
        if not self.current_data: return
        d = self.current_data
        try: y_s, y_e = float(self.roi_pfy_start.text()), float(self.roi_pfy_end.text()); x_s, x_e = float(self.roi_xes_start.text()), float(self.roi_xes_end.text())
        except ValueError: return
        if self.selector: self.selector.extents = (x_s, x_e, y_s, y_e)
        
        unit = "Flux-Corrected Counts" if self.count_toggle.isChecked() else "Normalized Intensity"
        iy_i, iy_f = np.argmin(np.abs(d['y'] - y_s)), np.argmin(np.abs(d['y'] - y_e))
        ix_i, ix_f = np.argmin(np.abs(d['x'] - x_s)), np.argmin(np.abs(d['x'] - x_e))

        # PFY: Multiplied by constant initial I0
        if self.count_toggle.isChecked():
            # Apply cumulative scale multiplier relative to the normalized sum
            raw_pfy = np.sum(d['z_norm'][:, min(iy_i, iy_f):max(iy_i, iy_f)], axis=1)
            # Find the scaling ratio from summed folders
            self.current_pfy_y = raw_pfy * d['i0_scale']
        else:
            self.current_pfy_y = np.sum(d['z_norm'][:, min(iy_i, iy_f):max(iy_i, iy_f)], axis=1)
        self.current_pfy_x = d['x']

        # XES: Multiplied by energy-dependent I0 array
        source_z = d['z_flux'] if self.count_toggle.isChecked() else d['z_norm']
        self.current_xes_x = np.sum(source_z[min(ix_i, ix_f):max(ix_i, ix_f), :], axis=0)
        self.current_xes_y = d['y']

        if self.pfy_line is None or self.pfy_line not in self.ax_pfy.lines: 
            self.pfy_line, = self.ax_pfy.plot(self.current_pfy_x, self.current_pfy_y, color='lime')
        else: self.pfy_line.set_ydata(self.current_pfy_y); self.ax_pfy.relim(); self.ax_pfy.autoscale_view(scalex=False, scaley=True)
        self.ax_pfy.set_title(f"PFY ({unit})")

        if self.xes_line is None or self.xes_line not in self.ax_xes.lines: 
            self.xes_line, = self.ax_xes.plot(self.current_xes_x, self.current_xes_y, color='cyan')
        else: self.xes_line.set_xdata(self.current_xes_x); self.ax_xes.relim(); self.ax_xes.autoscale_view(scalex=True, scaley=False)
        self.ax_xes.set_xlabel(unit); self.ax_xes.set_title("XES")
        
        for p in list(self.ax_pfy.patches): p.remove()
        self.ax_pfy.axvspan(x_s, x_e, color='cyan', alpha=0.1) 
        for p in list(self.ax_xes.patches): p.remove()
        self.ax_xes.axhspan(y_s, y_e, color='lime', alpha=0.1)
        for l in list(self.ax_map.lines): l.remove()
        self.ax_map.axhline(y_s, color='lime', linestyle='--', linewidth=1); self.ax_map.axhline(y_e, color='lime', linestyle='--', linewidth=1); self.ax_map.axvline(x_s, color='cyan', linestyle='--', linewidth=1); self.ax_map.axvline(x_e, color='cyan', linestyle='--', linewidth=1)
        self.canvas.draw()

    def export_2d_map(self):
        if self.current_data is None: QMessageBox.warning(self, "Export Error", "No map available."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save Summed RIXS Map", "summed_rixs_map.pkl.gz", "Gzip Pickle Files (*.pkl.gz)")
        if not path: return
        try:
            export_dict = {'rixs_sum': {'rixs_plane0_norm': np.array([self.current_data['z_norm']]), 'valid_nominal_mono': self.current_data['x'], 'bin_centers': self.current_data['y'], 'sample_name': 'Summed_Result', 'sample_id': 'SUM_ROI'}}
            with gzip.open(path, 'wb') as f: cPickle.dump(export_dict, f, protocol=4)
            QMessageBox.information(self, "Success", f"Summed RIXS map exported successfully.")
        except Exception as e: QMessageBox.critical(self, "Export Failed", str(e))

    def export_1d_data(self):
        if self.current_pfy_x is None: QMessageBox.warning(self, "Export Error", "No curves available."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save Integrated 1D Curves", "integrated_curves.txt", "Text Files (*.txt)")
        if not path: return
        pfy_roi = f"[{self.roi_pfy_start.text()}, {self.roi_pfy_end.text()}]"; xes_roi = f"[{self.roi_xes_start.text()}, {self.roi_xes_end.text()}]"
        is_scaled = "Flux-Corrected (I0 ratio scaling applied)." if self.count_toggle.isChecked() else "Normalized Intensity (No scaling)."
        try:
            with open(path.replace(".txt", "_pfy.txt"), 'w') as f:
                f.write(f"# PFY Integrated over Emission ROI: {pfy_roi} eV\n# {is_scaled}\n# Incident_Energy(eV)\tIntensity\n")
                for x, y in zip(self.current_pfy_x, self.current_pfy_y): f.write(f"{x:.6f}\t{y:.6f}\n")
            with open(path.replace(".txt", "_xes.txt"), 'w') as f:
                f.write(f"# XES Integrated over Excitation ROI: {xes_roi} eV\n# {is_scaled}\n# Emission_Energy(eV)\tIntensity\n")
                for y, x in zip(self.current_xes_y, self.current_xes_x): f.write(f"{y:.6f}\t{x:.6f}\n")
            QMessageBox.information(self, "Success", f"1D data exported successfully.")
        except Exception as e: QMessageBox.critical(self, "Export Failed", str(e))

    def update_color_scale(self):
        if not self.map_img: return
        raw = self.map_img.get_array(); d_min, d_max = np.min(raw), np.max(raw); vmin, vmax = d_min + (d_max - d_min) * (self.slider_min.value()/100), d_min + (d_max - d_min) * (self.slider_max.value()/100); self.map_img.set_clim(vmin, vmax if vmax > vmin else vmin + 1e-6); self.canvas.draw()

    def update_color_map(self, cmap_name):
        if self.map_img: self.map_img.set_cmap(cmap_name); self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv); window = RIXSGui(); window.show(); getattr(app, EXEC_FUNC)()