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
    from PyQt6 import QtWidgets, QtCore, QtGui
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QFileDialog, QTreeWidget, QTreeWidgetItem, 
                                 QAbstractItemView, QLabel, QLineEdit, QTextEdit, QSplitter)
    from PyQt6.QtCore import Qt
    matplotlib.use('QtAgg') 
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    EXEC_FUNC = "exec"
except ImportError:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QFileDialog, QTreeWidget, QTreeWidgetItem, 
                                 QAbstractItemView, QLabel, QLineEdit, QTextEdit, QSplitter)
    from PyQt5.QtCore import Qt
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    EXEC_FUNC = "exec_"

class RIXSGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RIXS Analyzer: Tree View & Summation")
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

        sidebar.addWidget(QLabel("Data Hierarchy (Ctrl+Click to sum):"))
        
        # CHANGED: Using QTreeWidget for hierarchy
        self.scan_tree = QTreeWidget()
        self.scan_tree.setHeaderLabel("Files / Scans")
        self.scan_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # Optional: Connect selection change if you want auto-plotting (currently manual via buttons)
        # self.scan_tree.itemSelectionChanged.connect(self.plot_rixs_map) 
        sidebar.addWidget(self.scan_tree)

        # --- ROI Controls ---
        sidebar.addWidget(QLabel("ROI Start (Emission):"))
        self.roi_start = QLineEdit("0")
        sidebar.addWidget(self.roi_start)
        
        sidebar.addWidget(QLabel("ROI End (Emission):"))
        self.roi_end = QLineEdit("1000")
        sidebar.addWidget(self.roi_end)

        self.plot_map_btn = QPushButton("Plot Summed RIXS Map")
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
        self.cmd_input.setPlaceholderText("Command line section")
        self.cmd_input.returnPressed.connect(self.execute_cmd)
        
        cli_widget = QWidget()
        cli_layout = QVBoxLayout(cli_widget)
        cli_layout.addWidget(QLabel("Console:"))
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
        if not files: return
        
        for f in files:
            try:
                # Logic from pkl2dict.py
                with gzip.GzipFile(f, "rb") as gz:
                    try:
                        data = cPickle.load(gz)
                    except UnicodeDecodeError:
                        data = cPickle.load(gz, encoding='latin1') 
                
                fname = os.path.basename(f)
                self.raw_data[fname] = data
                
                # CHANGED: Build Tree Hierarchy
                file_item = QTreeWidgetItem(self.scan_tree)
                file_item.setText(0, fname)
                file_item.setExpanded(True) # Auto-expand to show scans
                
                for scan_name in data.keys():
                    if isinstance(data[scan_name], dict):
                        scan_item = QTreeWidgetItem(file_item)
                        scan_item.setText(0, scan_name)
                        # We can store identifying data in the item itself to make retrieval easier
                        scan_item.setData(0, Qt.ItemDataRole.UserRole, (fname, scan_name))
                        
                self.console.append(f"Loaded: {fname}")
            except Exception as e:
                self.console.append(f"Error loading {f}: {e}")

    def get_selected_data(self):
        """
        Iterates over all selected items, sums their Z-matrices, 
        and returns the combined data.
        """
        selected_items = self.scan_tree.selectedItems()
        if not selected_items: return None

        summed_z = None
        common_x = None
        common_y = None
        count = 0

        for item in selected_items:
            # Skip top-level file items (they don't have tuple data in UserRole)
            user_data = item.data(0, Qt.ItemDataRole.UserRole)
            if not user_data: 
                continue 
            
            fname, sname = user_data
            rixs = self.raw_data[fname][sname]

            # CHANGED: Use the user-specified logic for Z key
            z_key = 'rixs_plane1_norm' if 'rixs_plane1_norm' in rixs else 'rixs_plane0_norm'
            x_key = 'valid_nominal_mono' if 'valid_nominal_mono' in rixs else 'nominal_mono'
            
            # Logic: Sum over scans (axis=0) for this specific file entry
            current_z = np.sum(rixs[z_key], axis=0)
            current_x = rixs[x_key]
            current_y = rixs['bin_centers']

            # Initialize accumulators with the first valid selection
            if summed_z is None:
                summed_z = current_z
                common_x = current_x
                common_y = current_y
            else:
                # Basic check to ensure we are summing compatible maps
                if current_z.shape == summed_z.shape:
                    summed_z += current_z
                else:
                    self.console.append(f"Warning: Skipping {sname} due to shape mismatch {current_z.shape} vs {summed_z.shape}")
            count += 1
        
        if summed_z is None: return None

        self.console.append(f"Summed {count} scans.")
        return {'x': common_x, 'y': common_y, 'z': summed_z}

    def plot_rixs_map(self):
        data = self.get_selected_data()
        if not data: return
        
        self.ax.clear()
        # Transpose Z for pcolor as per plotting_examples.py
        self.ax.pcolor(data['x'], data['y'], data['z'].T, shading='auto')
        self.ax.set_xlabel("Excitation Energy (eV)")
        self.ax.set_ylabel("Emission Energy (eV)")
        self.ax.set_title("Summed RIXS Map")
        self.canvas.draw()

    def plot_pfy(self):
        data = self.get_selected_data()
        if not data: return
        try:
            r_start = float(self.roi_start.text())
            r_end = float(self.roi_end.text())
            
            # ROI logic
            idx_i = np.argmin(np.abs(data['y'] - r_start))
            idx_f = np.argmin(np.abs(data['y'] - r_end))
            
            pfy = np.sum(data['z'][:, min(idx_i, idx_f):max(idx_i, idx_f)], axis=1)
            
            self.ax.clear()
            self.ax.plot(data['x'], pfy)
            self.ax.set_xlabel("Excitation Energy (eV)")
            self.ax.set_ylabel("Integrated Intensity (PFY)")
            self.ax.set_title("Summed PFY")
            self.canvas.draw()
        except Exception as e:
            self.console.append(f"ROI Error: {e}")

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
    getattr(app, EXEC_FUNC)()