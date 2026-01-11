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
    import pickle as cPickle # Handle py3 compatibility

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

class RIXSGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RIXS Analyzer: ROI & Integration")
        self.setGeometry(100, 100, 1400, 900)
        self.raw_data = {}
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Sidebar ---
        sidebar = QVBoxLayout()
        self.load_btn = QPushButton("Load .pkl.gz Files")
        self.load_btn.clicked.connect(self.load_files)
        sidebar.addWidget(self.load_btn)

        sidebar.addWidget(QLabel("Scans (Hold Ctrl to select multiple):"))
        self.scan_list = QListWidget()
        self.scan_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.scan_list.itemSelectionChanged.connect(self.update_plot)
        sidebar.addWidget(self.scan_list)

        # --- ROI Controls (New Integration Features) ---
        sidebar.addWidget(QLabel("ROI Start (Emission):"))
        self.roi_start = QLineEdit("0")
        sidebar.addWidget(self.roi_start)
        
        sidebar.addWidget(QLabel("ROI End (Emission):"))
        self.roi_end = QLineEdit("1000")
        sidebar.addWidget(self.roi_end)

        self.plot_map_btn = QPushButton("Plot RIXS Map")
        self.plot_map_btn.clicked.connect(self.plot_rixs_map)
        sidebar.addWidget(self.plot_map_btn)

        self.plot_pfy_btn = QPushButton("Plot PFY (Integrated)")
        self.plot_pfy_btn.clicked.connect(self.plot_pfy)
        sidebar.addWidget(self.plot_pfy_btn)

        # --- Plotting Area ---
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        # --- CLI & Console ---
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.cmd_input = QLineEdit()
        self.cmd_input.setPlaceholderText("Command line section for data handling")
        self.cmd_input.returnPressed.connect(self.execute_cmd)
        
        cli_widget = QWidget()
        cli_layout = QVBoxLayout(cli_widget)
        cli_layout.addWidget(QLabel("Console / CLI:"))
        cli_layout.addWidget(self.console)
        cli_layout.addWidget(self.cmd_input)

        # Layout Assembly
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.addWidget(self.canvas)
        v_splitter.addWidget(cli_widget)

        main_layout.addLayout(sidebar, 1)
        main_layout.addWidget(v_splitter, 4)

    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open RIXS Data")
        for f in files:
            # logic derived from pkl2dict.py
            try:
                with gzip.GzipFile(f, "rb") as gz:
                    try:
                        data = cPickle.load(gz)
                    except UnicodeDecodeError:
                        data = cPickle.load(gz, encoding='latin1') # likely python2 saved
                
                fname = os.path.basename(f)
                self.raw_data[fname] = data
                for scan_name in data.keys():
                    if isinstance(data[scan_name], dict):
                        self.scan_list.addItem(f"{fname} | {scan_name}")
                self.console.append(f"# gzipped {fname} loaded")
            except Exception as e:
                self.console.append(f"Error loading {f}: {e}")

    def get_selected_data(self):
        """Helper to extract data using rules from plotting_examples.py"""
        items = self.scan_list.selectedItems()
        if not items: return None
        f_key, s_key = items[0].text().split(" | ")
        rixs = self.raw_data[f_key][s_key]
        
        # Follows plotting_examples.py logic
        # x: excitation energy, y: emission axis (TES), z: intensity plane
        x_key = 'valid_nominal_mono' if 'valid_nominal_mono' in rixs else 'nominal_mono'
        # z_key = 'rixs_plane1_norm' if 'valid_nominal_mono' in rixs else 'rixs_plane0_norm'
        z_key = 'rixs_plane1_norm' if 'rixs_plane1_norm' in rixs else 'rixs_plane0_norm'
        
        return {
            'x': rixs[x_key],
            'y': rixs['bin_centers'],
            'z': np.sum(rixs[z_key], axis=0) # sum over different scans
        }

    def update_plot(self):
        # Placeholder to handle selection changes if needed
        pass

    def plot_rixs_map(self):
        """Plot rixs maps as defined in examples"""
        data = self.get_selected_data()
        if not data: return
        self.ax.clear()
        # plot rixs map using pcolor
        self.ax.pcolor(data['x'], data['y'], data['z'].T, shading='auto')
        self.ax.set_xlabel("Excitation Energy (eV)")
        self.ax.set_ylabel("Emission axis (TES)")
        self.canvas.draw()

    def plot_pfy(self):
        """Generate pfy from ROI integration"""
        data = self.get_selected_data()
        if not data: return
        try:
            r_start = float(self.roi_start.text())
            r_end = float(self.roi_end.text())
            
            # Find emission indices for roi_i and roi_f
            idx_i = np.argmin(np.abs(data['y'] - r_start))
            idx_f = np.argmin(np.abs(data['y'] - r_end))
            
            # Sum between roi_i and roi_f
            pfy = np.sum(data['z'][:, min(idx_i, idx_f):max(idx_i, idx_f)], axis=1)
            
            self.ax.clear()
            self.ax.plot(data['x'], pfy)
            self.ax.set_xlabel("Excitation Energy (eV)")
            self.ax.set_ylabel("Integrated Intensity (PFY)")
            self.canvas.draw()
        except Exception as e:
            self.console.append(f"ROI Integration Error: {e}")

    def execute_cmd(self):
        cmd = self.cmd_input.text()
        try:
            exec(cmd, {'self': self, 'np': np, 'plt': plt})
            self.console.append(f">>> {cmd}")
        except Exception as e:
            self.console.append(f"CLI Error: {e}")
        self.cmd_input.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RIXSGui()
    window.show()
    getattr(app, EXEC_FUNC)() # Compatibility for exec vs exec_