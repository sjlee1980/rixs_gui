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
        self.setWindowTitle("RIXS Analyzer: Perfected Math & Overlay")
        self.setGeometry(50, 50, 1600, 1000)
        self.raw_data = {}
        
        self.current_data = None
        self.map_img = None
        self.selector = None 
        self.cax = None      
        
        # Overlay and Legend state
        self.held_pfy = [] 
        self.held_xes = [] 
        self.color_cycle = ['lime', 'orange', 'magenta', 'cyan', 'yellow', 'red', 'white']
        self.current_color_idx = 0
        
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

        self.count_toggle = QCheckBox("Show Flux-Corrected Counts")
        self.count_toggle.clicked.connect(self.on_toggle_counts)
        sidebar.addWidget(self.count_toggle)

        self.hold_toggle = QCheckBox("Hold Curves (Overlay)")
        self.hold_toggle.toggled.connect(self.manage_overlay_state)
        sidebar.addWidget(self.hold_toggle)

        self.clear_overlay_btn = QPushButton("Clear Overlay")
        self.clear_overlay_btn.clicked.connect(self.clear_overlay)
        sidebar.addWidget(self.clear_overlay_btn)

        sidebar.addSpacing(10)
        
        self.save_map_btn = QPushButton("Save Summed RIXS Map (.pkl.gz)")
        self.save_map_btn.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold; padding: 5px;")
        self.save_map_btn.clicked.connect(self.export_2d_map)
        sidebar.addWidget(self.save_map_btn)

        self.export_btn = QPushButton("Save Integrated 1D Curves (.txt)")
        self.export_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 5px;")
        self.export_btn.clicked.connect(self.export_1d_data)
        sidebar.addWidget(self.export_btn)
        
        sidebar.addSpacing(10)

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
        self.gs = gridspec.GridSpec(2, 3, width_ratios=[4, 1.5, 0.2], height_ratios=[1, 4], wspace=0.35, hspace=0.35)
        
        self.ax_pfy = self.figure.add_subplot(self.gs[0, 0])
        self.ax_leg = self.figure.add_subplot(self.gs[0, 1])
        self.ax_leg.axis('off')
        
        self.ax_map = self.figure.add_subplot(self.gs[1, 0], sharex=self.ax_pfy)
        self.ax_xes = self.figure.add_subplot(self.gs[1, 1], sharey=self.ax_map)
        self.cax = self.figure.add_subplot(self.gs[1, 2])
        
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

    def on_toggle_counts(self):
        self.clear_overlay()

    def manage_overlay_state(self, checked):
        if not checked: self.clear_overlay()
        else:
            self.capture_current_to_overlay()
            self.update_integrations()

    def capture_current_to_overlay(self):
        if self.current_data is None: return
        selected = self.scan_tree.selectedItems()
        label = selected[0].text(0) if selected else "Active"
        
        iy_i, iy_f = self.get_integration_indices()
        ix_i, ix_f = self.get_excitation_indices()
        d = self.current_data

        # MATH FIX: Strictly multiply the raw PFY shape by a single scalar constant
        raw_pfy = np.sum(d['z_norm'][:, min(iy_i, iy_f):max(iy_i, iy_f)], axis=1)
        if self.count_toggle.isChecked():
            pfy0_flux = np.sum(d['z_flux'][0, min(iy_i, iy_f):max(iy_i, iy_f)])
            y_pfy = raw_pfy * (pfy0_flux / raw_pfy[0]) if raw_pfy[0] != 0 else raw_pfy
        else:
            y_pfy = raw_pfy

        source_z = d['z_flux'] if self.count_toggle.isChecked() else d['z_norm']
        x_xes = np.sum(source_z[min(ix_i, ix_f):max(ix_i, ix_f), :], axis=0)

        if not self.held_pfy or not np.array_equal(self.held_pfy[-1][1], y_pfy):
            color = self.color_cycle[self.current_color_idx % len(self.color_cycle)]
            self.held_pfy.append((d['x'], y_pfy, color, label))
            self.held_xes.append((x_xes, d['y'], color, label))
            self.current_color_idx += 1

    def clear_overlay(self):
        self.held_pfy, self.held_xes, self.current_color_idx = [], [], 0
        self.ax_pfy.clear(); self.ax_xes.clear(); self.ax_leg.clear(); self.ax_leg.axis('off')
        self.on_tree_selection()

    def get_summed_data(self):
        selected_items = self.scan_tree.selectedItems()
        if not selected_items: return None
        szn, cx, cy, szf = None, None, None, None
        
        for folder_item in selected_items:
            ud = folder_item.data(0, USER_ROLE)
            if not ud: continue
            fn, sn = ud; rixs = self.raw_data[fn][sn]
            raw_m, norm_m = np.sum(rixs['rixs_plane0'], axis=0), np.sum(rixs['rixs_plane0_norm'], axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                i0_arr = np.nan_to_num(np.nanmean(raw_m / norm_m, axis=1), nan=1.0)
            
            for i in range(folder_item.childCount()):
                sc = folder_item.child(i)
                if sc.checkState(0) == Qt.CheckState.Checked:
                    z_key = 'rixs_plane1_norm' if 'rixs_plane1_norm' in rixs else 'rixs_plane0_norm'
                    zn = np.sum(rixs[z_key], axis=0); x, y = rixs['valid_nominal_mono' if 'valid_nominal_mono' in rixs else 'nominal_mono'], rixs['bin_centers']
                    zf = zn * i0_arr[:, np.newaxis]
                    
                    if szn is None: szn, cx, cy, szf = zn.copy(), x, y, zf.copy()
                    elif zn.shape == szn.shape: szn += zn; szf += zf
                    
        return {'x': cx, 'y': cy, 'z_norm': szn, 'z_flux': szf} if szn is not None else None

    def on_tree_selection(self):
        QApplication.processEvents()
        new_data = self.get_summed_data()
        if not new_data: 
            self.ax_map.clear(); self.ax_pfy.clear(); self.ax_xes.clear(); self.ax_leg.clear(); self.ax_leg.axis('off'); self.canvas.draw(); return
        
        self.current_data = new_data; d = self.current_data; x_min, x_max = np.min(d['x']), np.max(d['x']); y_min, y_max = np.min(d['y']), np.max(d['y'])
        
        if not self.roi_pfy_start.text(): self.roi_pfy_start.setText(f"{y_min:.2f}"); self.roi_pfy_end.setText(f"{y_max:.2f}")
        if not self.roi_xes_start.text(): self.roi_xes_start.setText(f"{x_min:.2f}"); self.roi_xes_end.setText(f"{x_max:.2f}")

        try: ys, ye = float(self.roi_pfy_start.text()), float(self.roi_pfy_end.text()); xs, xe = float(self.roi_xes_start.text()), float(self.roi_xes_end.text())
        except ValueError: ys, ye, xs, xe = y_min, y_max, x_min, x_max

        self.ax_map.clear()
        disp_z = d['z_flux'] if self.count_toggle.isChecked() else d['z_norm']
        self.map_img = self.ax_map.pcolormesh(d['x'], d['y'], disp_z.T, shading='auto', cmap=self.cmap_combo.currentText())
        self.ax_map.set_xlabel("Incident Energy (eV)"); self.ax_map.set_ylabel("Emission Energy (eV)")
        self.ax_map.set_xlim(x_min, x_max); self.ax_map.set_ylim(y_min, y_max)
        
        if self.selector: self.selector.set_active(False)
        self.selector = RectangleSelector(self.ax_map, self.on_select_roi, useblit=True, button=[1], minspanx=5, minspany=5, interactive=True); self.selector.extents = (xs, xe, ys, ye)
        
        if self.hold_toggle.isChecked(): self.capture_current_to_overlay()
        
        self.cax.clear(); self.figure.colorbar(self.map_img, cax=self.cax); self.update_color_scale(); self.update_integrations()

    def get_integration_indices(self):
        d = self.current_data
        try: y_s, y_e = float(self.roi_pfy_start.text()), float(self.roi_pfy_end.text())
        except: y_s, y_e = np.min(d['y']), np.max(d['y'])
        return np.argmin(np.abs(d['y'] - y_s)), np.argmin(np.abs(d['y'] - y_e))

    def get_excitation_indices(self):
        d = self.current_data
        try: x_s, x_e = float(self.roi_xes_start.text()), float(self.roi_xes_end.text())
        except: x_s, x_e = np.min(d['x']), np.max(d['x'])
        return np.argmin(np.abs(d['x'] - x_s)), np.argmin(np.abs(d['x'] - x_e))

    def on_select_roi(self, eclick, erelease):
        x1, x2 = sorted([eclick.xdata, erelease.xdata]); y1, y2 = sorted([eclick.ydata, erelease.ydata])
        self.roi_xes_start.setText(f"{x1:.2f}"); self.roi_xes_end.setText(f"{x2:.2f}"); self.roi_pfy_start.setText(f"{y1:.2f}"); self.roi_pfy_end.setText(f"{y2:.2f}")
        self.update_integrations()

    def update_integrations_from_text(self): self.update_integrations()

    def update_integrations(self):
        if not self.current_data: return
        d = self.current_data
        iy_i, iy_f = self.get_integration_indices()
        ix_i, ix_f = self.get_excitation_indices()
        unit = "Flux-Corrected Counts" if self.count_toggle.isChecked() else "Normalized Intensity"
        
        # MATH FIX: Constant scalar logic for PFY
        raw_pfy = np.sum(d['z_norm'][:, min(iy_i, iy_f):max(iy_i, iy_f)], axis=1)
        if self.count_toggle.isChecked():
            pfy0_flux = np.sum(d['z_flux'][0, min(iy_i, iy_f):max(iy_i, iy_f)])
            cur_pfy_y = raw_pfy * (pfy0_flux / raw_pfy[0]) if raw_pfy[0] != 0 else raw_pfy
        else:
            cur_pfy_y = raw_pfy

        source_z = d['z_flux'] if self.count_toggle.isChecked() else d['z_norm']
        cur_xes_x = np.sum(source_z[min(ix_i, ix_f):max(ix_i, ix_f), :], axis=0)

        self.ax_pfy.clear(); self.ax_xes.clear(); self.ax_leg.clear(); self.ax_leg.axis('off')
        active_label = self.scan_tree.selectedItems()[0].text(0) if self.scan_tree.selectedItems() else "Active"

        # Suppress duplicate legend items for active folder
        for px, py, col, lbl in self.held_pfy:
            if lbl != active_label: self.ax_pfy.plot(px, py, color=col, alpha=0.6, linestyle='--', label=f"{lbl}")
        for xx, xy, col, lbl in self.held_xes:
            if lbl != active_label: self.ax_xes.plot(xx, xy, color=col, alpha=0.6, linestyle='--')

        act_col = self.color_cycle[self.current_color_idx % len(self.color_cycle)]
        self.ax_pfy.plot(d['x'], cur_pfy_y, color=act_col, linewidth=2, label=active_label)
        self.ax_xes.plot(cur_xes_x, d['y'], color=act_col, linewidth=2)

        if self.hold_toggle.isChecked() or self.held_pfy:
            h, l = self.ax_pfy.get_legend_handles_labels()
            self.ax_leg.legend(h, l, loc='center', fontsize='small', frameon=True)

        self.ax_pfy.set_title(f"PFY ({unit})"); self.ax_xes.set_xlabel(unit); self.ax_xes.set_title("XES")
        self.ax_pfy.set_xlim(np.min(d['x']), np.max(d['x'])); self.ax_xes.set_ylim(np.min(d['y']), np.max(d['y']))
        self.ax_pfy.relim(); self.ax_pfy.autoscale_view(); self.ax_xes.relim(); self.ax_xes.autoscale_view()
        
        try: xs, xe = float(self.roi_xes_start.text()), float(self.roi_xes_end.text()); ys, ye = float(self.roi_pfy_start.text()), float(self.roi_pfy_end.text())
        except: xs, xe, ys, ye = np.min(d['x']), np.max(d['x']), np.min(d['y']), np.max(d['y'])
        
        self.ax_pfy.axvspan(xs, xe, color='cyan', alpha=0.1); self.ax_xes.axhspan(ys, ye, color='lime', alpha=0.1)
        for line in list(self.ax_map.lines): line.remove()
        self.ax_map.axhline(ys, color='lime', linestyle='--', linewidth=1); self.ax_map.axhline(ye, color='lime', linestyle='--', linewidth=1)
        self.ax_map.axvline(xs, color='cyan', linestyle='--', linewidth=1); self.ax_map.axvline(xe, color='cyan', linestyle='--', linewidth=1)
        self.canvas.draw()

    def export_2d_map(self):
        if not self.current_data: return
        path, _ = QFileDialog.getSaveFileName(self, "Save RIXS Map", "map.pkl.gz", "Gzip Pickle (*.pkl.gz)")
        if not path: return
        try:
            export_dict = {'rixs_sum': {'rixs_plane0_norm': np.array([self.current_data['z_norm']]), 'valid_nominal_mono': self.current_data['x'], 'bin_centers': self.current_data['y'], 'sample_name': 'Summed', 'sample_id': 'SUM'}}
            with gzip.open(path, 'wb') as f: cPickle.dump(export_dict, f, protocol=4)
        except Exception as e: QMessageBox.critical(self, "Export Failed", str(e))

    def export_1d_data(self):
        if not self.current_data: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Integrated Curves", "curves.txt", "Text Files (*.txt)")
        if not path: return
        try:
            with open(path.replace(".txt", "_pfy.txt"), 'w') as f:
                f.write(f"# PFY Data\n# Energy(eV)\tIntensity\n")
                d = self.current_data; iy_i, iy_f = self.get_integration_indices()
                raw_pfy = np.sum(d['z_norm'][:, min(iy_i, iy_f):max(iy_i, iy_f)], axis=1)
                
                # Apply same scalar math on export
                if self.count_toggle.isChecked():
                    pfy0_flux = np.sum(d['z_flux'][0, min(iy_i, iy_f):max(iy_i, iy_f)])
                    y_pfy = raw_pfy * (pfy0_flux / raw_pfy[0]) if raw_pfy[0] != 0 else raw_pfy
                else:
                    y_pfy = raw_pfy

                for x, y in zip(d['x'], y_pfy): f.write(f"{x:.6f}\t{y:.6f}\n")
            QMessageBox.information(self, "Success", "PFY data exported.")
        except Exception as e: QMessageBox.critical(self, "Export Failed", str(e))

    def update_color_scale(self):
        if not self.map_img: return
        raw = self.map_img.get_array(); dmin, dmax = np.min(raw), np.max(raw); vmin, vmax = dmin + (dmax-dmin)*(self.slider_min.value()/100), dmin + (dmax-dmin)*(self.slider_max.value()/100); self.map_img.set_clim(vmin, vmax if vmax > vmin else vmin+1e-6); self.canvas.draw()

    def update_color_map(self, name):
        if self.map_img: self.map_img.set_cmap(name); self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv); window = RIXSGui(); window.show(); getattr(app, EXEC_FUNC)()