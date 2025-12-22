#!/usr/bin/env python3
# Fast 2×2 (or 1–4 panes) point cloud viewer — PyQt5 + VisPy
# • Recursively loads .ply/.pcd from subfolders
# • Pairs (0↔1) and (2↔3) by basename intersection, lexicographically
# • Left/Right arrow + buttons for navigation
# • Point size slider (fast, continuous)
# • Titles as QLabels (no clipping)
# • Linked orthographic cameras, top-down XY view

import os
import sys
import glob
import argparse
import json
from functools import lru_cache
from typing import List, Tuple, Dict
from pathlib import Path

from appdirs import user_data_dir
import numpy as np
import open3d as o3d
from natsort import natsorted
from PyQt5 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor
import pyvista as pv

# ────────── DEFAULT FOLDERS (yours) ──────────
DEFAULT_FOLDERS = []
# [
#     r"F:\Datasets\SyntheticCRACK\Paper 2b - A graph-based 3D point clouds (Blend Comparision)\Colored-AlphaBlend",
#     r"F:\Datasets\SyntheticCRACK\Paper 2b - A graph-based 3D point clouds (Blend Comparision)\Groundtruth-AlphaBlend",
#     r"F:\Datasets\SyntheticCRACK\Paper 2b - A graph-based 3D point clouds (Blend Comparision)\Colored-Shading",
#     r"F:\Datasets\SyntheticCRACK\Paper 2b - A graph-based 3D point clouds (Blend Comparision)\Groundtruth-Shading",
# ]

# ────────── Constants ──────────
POINT_SIZE = 2.8  # initial point size in pixels
TIME_PAUSE = 1.5  # seconds between frames during Loop
APP_ICON = "app.ico"  # place app.ico next to this script or give an absolute path
STATE_JSON = "viewer_state.json"

# create (and ensure) the app-specific folder under user AppData
APP_NAME = "Point Cloud Compare"
state_dir = Path(user_data_dir(APP_NAME, appauthor=False))
state_dir.mkdir(parents=True, exist_ok=True)
STATE_JSON = os.path.join(state_dir, STATE_JSON)

# ────────── File discovery & alignment ──────────

def list_pc_files(folder: str) -> List[str]:
    """Recursively collect .ply/.pcd under folder; stable lexicographic order."""
    pats = ("**/*.ply", "**/*.PLY", "**/*.pcd", "**/*.PCD")
    pats = ("**/*.ply", "**/*.PLY", "**/*.pcd", "**/*.PCD")
    files = []
    for pat in pats:
        files.extend(glob.glob(os.path.join(folder, pat), recursive=True))
    # Sort by relative path (case-insensitive), then full path
    return natsorted(files, key=lambda p: os.path.relpath(p, folder).lower())

def map_by_basename(paths: List[str]) -> Dict[str, str]:
    """Case-insensitive basename map to fullpath; last wins but order is sorted upstream."""
    m: Dict[str, str] = {}
    for p in paths:
        m[os.path.basename(p).lower()] = p
    return m

def align_pair(fA: str, fB: str) -> Tuple[List[str], List[str]]:
    """
    Align two folders by basename intersection (case-insensitive), lexicographically.
    Returns two equally-long lists in matching order.
    """
    A = list_pc_files(fA)
    B = list_pc_files(fB)
    if len(A) == 0 or len(B) == 0:
        return [], []
    mA = map_by_basename(A)
    mB = map_by_basename(B)
    common = natsorted(set(mA.keys()) & set(mB.keys()))
    return [mA[k] for k in common], [mB[k] for k in common]

def align_file_sets(folder_paths: List[str]) -> Tuple[List[List[str]], int]:
    """
    1 folder  -> that list
    2 folders -> pairwise aligned (0↔1)
    3 folders -> (0↔1) aligned; 2nd pair absent; truncate to len(pair1)
    4 folders -> (0↔1) and (2↔3) aligned; overall length = min(len(pair1), len(pair2))
    """
    k = len(folder_paths)
    if k == 0:
        raise RuntimeError("No folders provided.")

    if k == 1:
        L0 = list_pc_files(folder_paths[0])
        return [L0], len(L0)

    if k == 2:
        L0, L1 = align_pair(folder_paths[0], folder_paths[1])
        n = min(len(L0), len(L1))
        return [L0[:n], L1[:n]], n

    if k == 3:
        L0, L1 = align_pair(folder_paths[0], folder_paths[1])
        L2 = list_pc_files(folder_paths[2])
        n = min(len(L0), len(L1), len(L2))
        return [L0[:n], L1[:n], L2[:n]], n

    # k >= 4 → use first four
    L0, L1 = align_pair(folder_paths[0], folder_paths[1])
    L2, L3 = align_pair(folder_paths[2], folder_paths[3])
    n = min(len(L0), len(L1), len(L2), len(L3))
    return [L0[:n], L1[:n], L2[:n], L3[:n]], n

# ────────── Point cloud IO (cached) ──────────
@lru_cache(maxsize=96)
def load_cloud(path: str):
    """Load points + colors exactly as stored. No contrast/brightness tweaks."""
    pcd = o3d.io.read_point_cloud(path, print_progress=False)
    if pcd.is_empty():
        return np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32)

    pts = np.asarray(pcd.points, dtype=np.float32)

    if pcd.has_colors():
        cols = np.asarray(pcd.colors)
        cols = np.asarray(cols)  # ensure ndarray
        if cols.ndim != 2 or cols.shape[1] < 3:
            cols = np.full((pts.shape[0], 3), 0.5, np.float32)  # fallback mid-gray
        else:
            cols = cols[:, :3]
            # true-color: only normalize if clearly uint8
            if np.nanmax(cols) > 1.001:
                cols = cols.astype(np.float32) / 255.0
            else:
                cols = cols.astype(np.float32)
    else:
        cols = np.full((pts.shape[0], 3), 0.5, np.float32)  # no color in file

    cols = np.clip(cols, 0.0, 1.0)
    alpha = np.ones((cols.shape[0], 1), dtype=np.float32)
    cols = np.concatenate([cols, alpha], axis=1)
    return pts, cols

# ────────── PyVista canvas widget ──────────
class CloudPane(QtWidgets.QWidget):
    """QWidget wrapper: title + PyVista QtInteractor."""
    def __init__(self, title: str, shared_camera=None, parent=None):
        super().__init__(parent)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(4, 0, 4, 0)
        vbox.setSpacing(2)

        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setStyleSheet("color: #222;")
        vbox.addWidget(self.title_label, 0)

        # Create PyVista interactor
        self.plotter = QtInteractor(self)
        
        # ← IMPORTANT: do NOT let the interactor take keyboard focus
        self.plotter.setFocusPolicy(QtCore.Qt.NoFocus)
        try:
            # some versions expose the internal widget as .interactor
            self.plotter.interactor.setFocusPolicy(QtCore.Qt.NoFocus)
        except Exception:
            pass

        vbox.addWidget(self.plotter.interactor, 1)
        
        # Optional: smoother points/edges
        self.plotter.enable_anti_aliasing()

        # Keep references
        self._pos_full = None
        self._cols_full = None
        self.point_size = POINT_SIZE

        # Use same shared camera
        if shared_camera is not None:
            self.shared_camera = shared_camera
            self.plotter.renderer.SetActiveCamera(shared_camera)
        else:
            self.shared_camera = self.plotter.renderer.GetActiveCamera()
            self.shared_camera.SetParallelProjection(True)

        self.actor = None

    def set_title(self, text: str):
        self.title_label.setText(text)

    def set_camera(self, cam):
        self.shared_camera = cam
        self.shared_camera.SetParallelProjection(True)
        self.plotter.renderer.SetActiveCamera(self.shared_camera)

    def set_points(self, pts, cols, size_px):
        if pts is None or pts.size == 0:
            if self.actor:
                self.plotter.remove_actor(self.actor)
                self.actor = None
            return

        self._pos_full = pts
        self._cols_full = cols

        # Remove previous actor (do NOT clear renderer)
        if self.actor is not None:
            self.plotter.remove_actor(self.actor)

        cloud = pv.PolyData(pts)
        cloud["rgb"] = (cols[:, :3] * 255).astype(np.uint8)

        # Use spherical glyphs for nicer scaling
        self.actor = self.plotter.add_points(
            cloud,
            scalars="rgb",
            rgb=True,
            point_size=size_px,
            render_points_as_spheres=True,   # <-- was False
            reset_camera=False,   # <-- prevent VTK from auto-fitting per pane
            render=False,         # <-- do not render yet
        )
        
        prop = self.actor.GetProperty()
        prop.SetAmbient(1.0)
        prop.SetDiffuse(0.0)
        prop.SetSpecular(0.0)
        prop.LightingOff()

        self.point_size = float(size_px)


    def set_point_size(self, size_px):
        self.point_size = float(size_px)
        if self.actor is not None:
            self.actor.GetProperty().SetPointSize(self.point_size)
            self.plotter.render()
            
            
    def close(self):
        try:
            self.plotter.close()  # safely destroys VTK render window + interactor
        except Exception:
            pass


# ────────── Main window ──────────
class MainWindow(QtWidgets.QWidget):
    def __init__(self, folder_paths: List[str]):
        super().__init__()
        if os.path.exists(APP_ICON):
            self.setWindowIcon(QtGui.QIcon(APP_ICON))
        self.setWindowTitle("Point Cloud Compare")
        self.folders = [os.path.abspath(f) for f in folder_paths[:4]]
        self.idx = 0

        # Load from JSON if no folders given on CLI
        if not self.folders:
            self._load_state()
    
        # If no folders on startup, defer alignment until user clicks “Folders…”
        if self.folders:
            self.file_lists, self.num_frames = align_file_sets(self.folders)
            self.idx = max(0, min(self.idx, self.num_frames - 1)) if self.num_frames > 0 else 0
        else:
            self.file_lists, self.num_frames = [], 0

        self.point_size = POINT_SIZE

        # ── Top bar: [left zone] … center zone … [right zone]
        root = QtWidgets.QVBoxLayout(self)

        # use a 3-column grid so the center column is truly centered
        top = QtWidgets.QGridLayout()
        root.addLayout(top)

        # LEFT (frame + filename)
        left = QtWidgets.QHBoxLayout()
        # NEW: Folders… button (kept lightweight; selects 4 folders)
        self.btn_folders = QtWidgets.QPushButton("Folders…")
        self.btn_folders.setFixedWidth(84)
        self.btn_folders.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        left.addSpacing(12)
        left.addWidget(self.btn_folders)

        self.lbl_idx = QtWidgets.QLabel(self._idx_text())
        self.lbl_name = QtWidgets.QLabel("")
        self.lbl_name.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        left.addWidget(self.lbl_idx)
        left.addSpacing(12)
        left.addWidget(self.lbl_name)

        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems([
            "Top view (Ctrl+T)",
            "Bottom view (Ctrl+B)",
            "Front view (Ctrl+F)",
            "Back view (Ctrl+V)",
            "Left view (Ctrl+L)",
            "Right view (Ctrl+R)",
            "SW Isometric view (Ctrl+W)",
            "SE Isometric view (Ctrl+E)",
            "NW Isometric view (Ctrl+I)",
            "NE Isometric view (Ctrl+O)",
        ])
        self.btn_reset = QtWidgets.QPushButton("Reset (R)")
        
        # NEW: Loop toggle button (to the right of Reset)
        self.btn_loop = QtWidgets.QPushButton("Loop")
        self.btn_loop.setCheckable(True)
        
        # NEW: pause spin (seconds)
        self.spin_pause = QtWidgets.QDoubleSpinBox()
        self.spin_pause.setRange(0.00, 10.0)
        self.spin_pause.setSingleStep(0.05)
        self.spin_pause.setDecimals(2)
        self.spin_pause.setValue(0.80)               # default matches TIME_PAUSE below
        self.spin_pause.setSuffix(" s")
        self.spin_pause.setFixedWidth(72)
        self.spin_pause.setToolTip("Delay between frames while looping")
        self.spin_pause.setFocusPolicy(QtCore.Qt.ClickFocus)   # allow clicking to type
        self.spin_pause.setKeyboardTracking(False)             # commit on Enter/focus-out
    
        # CENTER (fixed-width optical center)
        center = QtWidgets.QHBoxLayout()
        center.setContentsMargins(0, 0, 0, 0)

        # ---- fixed-width container ----
        center_box = QtWidgets.QWidget()
        center_box.setFixedWidth(430)   # <<< key: tune once, stays perfect
        center_box_layout = QtWidgets.QHBoxLayout(center_box)
        center_box_layout.setContentsMargins(0, 0, 0, 0)
        center_box_layout.setSpacing(8)

        # --- left group: view + reset ---
        left_grp = QtWidgets.QHBoxLayout()
        left_grp.setSpacing(0)
        left_grp.addWidget(self.view_combo)
        left_grp.addWidget(self.btn_reset)

        # --- right group: loop + seconds ---
        right_grp = QtWidgets.QHBoxLayout()
        right_grp.setSpacing(0)
        right_grp.addWidget(self.btn_loop)
        right_grp.addWidget(self.spin_pause)

        # --- assemble ---
        center_box_layout.addLayout(left_grp)
        center_box_layout.addLayout(right_grp)

        center.addWidget(center_box, alignment=QtCore.Qt.AlignHCenter)

        # RIGHT (point size + slider + prev/next)
        right = QtWidgets.QHBoxLayout()
        right.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        # --- group: [Point size] [slider] [value]  (kept together)
        ps_widget = QtWidgets.QWidget()
        ps_row = QtWidgets.QHBoxLayout(ps_widget)
        ps_row.setContentsMargins(0, 0, 0, 0)
        ps_row.setSpacing(6)

        ps_row.addWidget(QtWidgets.QLabel("Point size"))

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(1, 80)                          # 1→0.1 … 80→8.0
        self.slider.setValue(int(self.point_size * 10))
        self.slider.setTracking(True)
        self.slider.setFixedWidth(140)
        self.slider.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                QtWidgets.QSizePolicy.Preferred)  # don't expand
        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        ps_row.addWidget(self.slider)

        self.lbl_size_val = QtWidgets.QLabel(f"{self.point_size:.1f}")
        self.lbl_size_val.setFixedWidth(28)                   # snug
        self.lbl_size_val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ps_row.addWidget(self.lbl_size_val)

        right.addWidget(ps_widget)                            # ← add the grouped row

        right.addSpacing(12)

        self.btn_prev = QtWidgets.QPushButton("◀ Prev")
        self.btn_next = QtWidgets.QPushButton("Next ▶")
        for b in (self.btn_prev, self.btn_next):
            b.setFixedWidth(86)
            b.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        right.addWidget(self.btn_prev)
        right.addWidget(self.btn_next)

        # --- place into the grid (center stays geometrically centered)
        top.addLayout(left,   0, 0, alignment=QtCore.Qt.AlignLeft)
        top.addLayout(center, 0, 1, alignment=QtCore.Qt.AlignHCenter)
        top.addLayout(right,  0, 2, alignment=QtCore.Qt.AlignRight)

        # keep side columns elastic; middle column fixed to its sizeHint
        top.setColumnStretch(0, 1)
        top.setColumnStretch(1, 0)
        top.setColumnStretch(2, 1)

        # ── Grid of panes
        self.grid = QtWidgets.QGridLayout()
        root.addLayout(self.grid, 1)
        # ensure equal space for 2×2
        self.grid.setRowStretch(0, 1)
        self.grid.setRowStretch(1, 1)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)
        
        # --- View shortcuts (Ctrl+T / Ctrl+B / Ctrl+I) ---
        sc_top = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+T"), self)
        sc_top.setContext(QtCore.Qt.ApplicationShortcut)
        sc_top.activated.connect(lambda: self.view_combo.setCurrentIndex(0))

        sc_bottom = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+B"), self)
        sc_bottom.setContext(QtCore.Qt.ApplicationShortcut)
        sc_bottom.activated.connect(lambda: self.view_combo.setCurrentIndex(1))
        
        sc_front = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+F"), self)
        sc_front.setContext(QtCore.Qt.ApplicationShortcut)
        sc_front.activated.connect(lambda: self.view_combo.setCurrentIndex(2))
        
        sc_back = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+V"), self)
        sc_back.setContext(QtCore.Qt.ApplicationShortcut)
        sc_back.activated.connect(lambda: self.view_combo.setCurrentIndex(3))
        
        sc_left = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+L"), self)
        sc_left.setContext(QtCore.Qt.ApplicationShortcut)
        sc_left.activated.connect(lambda: self.view_combo.setCurrentIndex(4))
        
        sc_right = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+R"), self)
        sc_right.setContext(QtCore.Qt.ApplicationShortcut)
        sc_right.activated.connect(lambda: self.view_combo.setCurrentIndex(5))

        sc_swiso = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+W"), self)
        sc_swiso.setContext(QtCore.Qt.ApplicationShortcut)
        sc_swiso.activated.connect(lambda: self.view_combo.setCurrentIndex(6))
        
        sc_seiso = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+E"), self)
        sc_seiso.setContext(QtCore.Qt.ApplicationShortcut)
        sc_seiso.activated.connect(lambda: self.view_combo.setCurrentIndex(7))
        
        sc_nwiso = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+I"), self)
        sc_nwiso.setContext(QtCore.Qt.ApplicationShortcut)
        sc_nwiso.activated.connect(lambda: self.view_combo.setCurrentIndex(8))
        
        sc_neiso = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+O"), self)
        sc_neiso.setContext(QtCore.Qt.ApplicationShortcut)
        sc_neiso.activated.connect(lambda: self.view_combo.setCurrentIndex(9))
        
        # --- Panes ---
        shared_cam = pv.Camera()
        shared_cam.SetParallelProjection(True)

        self.panes: List[CloudPane] = []
        titles = [os.path.basename(f) for f in self.folders]
        nviews = len(self.file_lists)

        rows = 1 if nviews <= 2 else 2
        cols = nviews if nviews <= 2 else 2

        for i in range(nviews):
            pane = CloudPane(titles[i], shared_camera=shared_cam, parent=self)
            self.panes.append(pane)
            self.grid.addWidget(pane, i // cols, i % cols)

        for p in self.panes:
            p.plotter.interactor.installEventFilter(self)

        # prevent arrow keys from landing on controls
        for w in (self.btn_prev, self.btn_next, self.btn_reset, self.view_combo, self.btn_loop):
            w.setFocusPolicy(QtCore.Qt.NoFocus)

        # ── Signals
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_next.clicked.connect(self.next_frame)
        self.slider.valueChanged.connect(self.on_size_changed)
        self.view_combo.currentIndexChanged.connect(self.on_view_mode_changed)
        self.btn_reset.clicked.connect(self.reset_view)

    
        # NEW: Loop timer + wiring
        self.loop_timer = QtCore.QTimer(self)
        self.loop_timer.setInterval(800)
        self.loop_timer.timeout.connect(self._advance_or_stop)
        self.btn_loop.toggled.connect(self.toggle_loop)

        def _apply_pause():
            self.loop_timer.setInterval(int(self.spin_pause.value() * 1000))
            self.setFocus(QtCore.Qt.OtherFocusReason)  # give Left/Right back to the window

        self.spin_pause.valueChanged.connect(lambda _: _apply_pause())   # arrows/wheel
        self.spin_pause.editingFinished.connect(_apply_pause)            # manual typing + Enter/tab/click-out

        self.btn_folders.clicked.connect(self.choose_folders)
        
        # Keyboard focus
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setFocus() 

        self.shared_cam = self._build_shared_camera("Top (XY)")
        for p in self.panes:
            p.set_camera(self.shared_cam)

        # Initial frame
        self.update_frame()

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Wheel:
            self._on_wheel(event, obj)
            return True   # block PyVista default zoom
        return super().eventFilter(obj, event)

    def _on_wheel(self, event, interactor):
        # wheel delta (+120, -120, etc.)
        delta = event.angleDelta().y()
        if delta == 0:
            return

        # zoom factor (AutoCAD-like smoothness)
        zoom_factor = 0.9 if delta > 0 else 1.1

        # get mouse position on widget
        pos = event.pos()

        # convert to world coordinates (using any pane that matches the interactor)
        for p in self.panes:
            if p.plotter.interactor is interactor:
                world_before = self._screen_to_world(p, pos.x(), pos.y())
                break
        else:
            return

        cam = self.shared_cam
        old_scale = cam.GetParallelScale()
        new_scale = old_scale * zoom_factor
        cam.SetParallelScale(new_scale)

        # get new world coordinate under same cursor
        world_after = self._screen_to_world(p, pos.x(), pos.y())

        if world_before is None or world_after is None:
            return

        # translation needed to keep point fixed under cursor
        shift = world_before - world_after

        # apply to camera position AND focal point
        cam.SetPosition(*(np.array(cam.GetPosition()) + shift))
        cam.SetFocalPoint(*(np.array(cam.GetFocalPoint()) + shift))

        cam.Modified()

        # apply to all panes
        for pane in self.panes:
            pane.plotter.renderer.ResetCameraClippingRange()
            pane.plotter.render()

    def _screen_to_world(self, pane, sx, sy):
        """Convert screen pixel → world coordinates for orthographic camera."""
        ren = pane.plotter.renderer
        if ren is None:
            return None

        # Convert Qt coords (top-left origin) to VTK coords (bottom-left origin)
        h = pane.plotter.window_size[1]
        sy_vtk = h - sy

        picker = ren.GetRenderWindow().GetInteractor().GetPicker()
        if picker is None:
            picker = vtk.vtkWorldPointPicker()
            ren.GetRenderWindow().GetInteractor().SetPicker(picker)

        if picker.Pick(sx, sy_vtk, 0, ren):
            return np.array(picker.GetPickPosition(), dtype=float)

        return None

    def _idx_text(self) -> str:
        return f"Frame {self.idx + 1} / {self.num_frames}"
    
    def _load_state(self):
        try:
            with open(STATE_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            # restore folders first (if exist and valid)
            if "folders" in data and isinstance(data["folders"], list):
                valid = [f for f in data["folders"] if isinstance(f, str) and os.path.isdir(f)]
                if valid:
                    self.folders = [os.path.abspath(f) for f in valid[:4]]
            # restore index (clamped after we know num_frames)
            self.idx = int(data.get("idx", 0))
        except Exception:
            pass

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(STATE_JSON), exist_ok=True)
            data = {
                "idx": int(self.idx),
                "folders": self.folders,     # NEW
            }
            with open(STATE_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def choose_folders(self):
        """Pick 1–4 folders at once (multi-select). No prompt at startup."""
        dlg = QtWidgets.QFileDialog(self, "Select 1–4 folders")
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)  # needed for multi-select on some platforms
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)

        # enable multi-selection in both the list and tree views
        for view in dlg.findChildren((QtWidgets.QListView, QtWidgets.QTreeView)):
            view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        if dlg.exec_():
            picked = dlg.selectedFiles()  # returns list of selected directories
            if picked:
                self.apply_new_folders(picked[:4])

    def apply_new_folders(self, folders):
        self.folders = [os.path.abspath(f) for f in folders[:4]]
        self.file_lists, self.num_frames = align_file_sets(self.folders)

        # --- rebuild panes if count changed (handles 0 → N and N → M)
        new_nviews = len(self.file_lists)

        # remove existing pane widgets from the grid
        for p in getattr(self, "panes", []):
            self.grid.removeWidget(p)
            p.setParent(None)
            p.deleteLater()

        self.panes = []
        titles = [os.path.basename(f) for f in self.folders]
        rows = 1 if new_nviews <= 2 else 2
        cols = new_nviews if new_nviews <= 2 else 2

        shared_cam = self.shared_cam  # reuse existing shared camera
        for i in range(new_nviews):
            pane = CloudPane(titles[i], shared_camera=shared_cam, parent=self)
            self.panes.append(pane)
            self.grid.addWidget(pane, i // cols, i % cols)

        # attach the shared orthographic camera to every pane
        for p in self.panes:
            p.set_camera(self.shared_cam)
    
        # keep idx in-bounds
        self.idx = 0 if self.num_frames == 0 else min(self.idx, self.num_frames - 1)

        # fit + draw or show placeholders
        if self.num_frames > 0:
            self.update_frame()
            self.reset_view()
        else:
            self.lbl_idx.setText("Frame 0 / 0")
            self.lbl_name.setText("No aligned files found")

    
    def _prompt_initial_folders(self):
        """Prompt for 1–4 folders at startup if none provided/loaded."""
        picked = []
        for i in range(4):
            start = os.path.expanduser("~")
            d = QtWidgets.QFileDialog.getExistingDirectory(self, f"Select folder {i+1} of 4 (Esc to stop)", start)
            if not d:
                break  # allow 1–4; stop when user cancels
            picked.append(d)
        return [os.path.abspath(f) for f in picked]

    def _build_shared_camera(self, mode: str):
        cam = pv.Camera()
        cam.SetParallelProjection(True)    # orthographic, like TurntableCamera fov=0
        return cam

    def toggle_loop(self, checked: bool):
        if checked and self.num_frames > 0:
            self.loop_timer.start()
        else:
            self.loop_timer.stop()

    def _advance_or_stop(self):
        if self.num_frames == 0:
            self.btn_loop.setChecked(False)
            return
        self.idx = (self.idx + 1) % self.num_frames
        self.update_frame()

    def _fit_all_to_data(self):
        mins, maxs = [], []
        for p in self.panes:
            if p._pos_full is None or p._pos_full.size == 0:
                continue
            mins.append(np.min(p._pos_full, axis=0))
            maxs.append(np.max(p._pos_full, axis=0))

        if not mins:
            return

        mn = np.min(np.vstack(mins), axis=0)
        mx = np.max(np.vstack(maxs), axis=0)

        cx, cy, cz = center = (mn + mx) * 0.5
        dx, dy, dz = mx - mn

        diag = np.linalg.norm([dx, dy, dz]) or 1.0
        dist = diag * 1.3      # comfortable camera distance
        scale = diag * 0.38    # matches VisPy VIEW_PAD = 0.75

        cam = self.shared_cam
        mode = self.view_combo.currentText()

        # always center camera on the data
        cam.SetFocalPoint(cx, cy, cz)

        if mode.startswith("Top"):
            # Top view: look DOWN -Z
            cam.SetPosition(cx, cy, cz + dist)
            cam.SetViewUp(0, 1, 0)
            cam.SetParallelProjection(True)

        elif mode.startswith("Bottom"):
            # Bottom view: look UP +Z
            cam.SetPosition(cx, cy, cz - dist)
            cam.SetViewUp(0, 1, 0)
            cam.SetParallelProjection(True)
            
        elif mode.startswith("Front"):
            # Front view: look +Y
            cam.SetPosition(cx, cy - dist, cz)
            cam.SetViewUp(0, 0, 1)
            cam.SetParallelProjection(True)
            
        elif mode.startswith("Back"):
            # Back view: look -Y
            cam.SetPosition(cx, cy + dist, cz)
            cam.SetViewUp(0, 0, 1)
            cam.SetParallelProjection(True)
            
        elif mode.startswith("Left"):
            # Left view: look -X
            cam.SetPosition(cx - dist, cy, cz)
            cam.SetViewUp(0, 0, 1)
            cam.SetParallelProjection(True)
            
        elif mode.startswith("Right"):
            # Right view: look +X
            cam.SetPosition(cx + dist, cy, cz)
            cam.SetViewUp(0, 0, 1)
            cam.SetParallelProjection(True)

        elif mode.startswith("SW"):
            # SOUTH-WEST isometric (-X, -Y, +Z)
            cam.SetPosition(cx - dist, cy - dist, cz + dist)
            cam.SetViewUp(0, 0, 1)
            cam.SetParallelProjection(True)
                        
        elif mode.startswith("SE"):
            # SOUTH-EAST isometric (+X, -Y, +Z)
            cam.SetPosition(cx + dist, cy - dist, cz + dist)
            cam.SetViewUp(0, 0, 1)
            cam.SetParallelProjection(True)
            
        elif mode.startswith("NW"):
            # NORTH-WEST isometric (-X, +Y, +Z)
            cam.SetPosition(cx - dist, cy + dist, cz + dist)
            cam.SetViewUp(0, 0, 1)
            cam.SetParallelProjection(True)
            
        elif mode.startswith("NE"):
            # NORTH-EAST isometric (+X, +Y, +Z)
            cam.SetPosition(cx + dist, cy + dist, cz + dist)
            cam.SetViewUp(0, 0, 1)
            cam.SetParallelProjection(True)

        cam.SetParallelProjection(True)
        cam.SetParallelScale(scale)
        cam.Modified()

        for p in self.panes:
            p.plotter.renderer.ResetCameraClippingRange()
            p.plotter.render()

    def reset_view(self):
        self._fit_all_to_data()
        self.setFocus(QtCore.Qt.OtherFocusReason)

    def on_view_mode_changed(self, idx: int):
        # mode already changed in combo; just refit camera & reattach
        self._fit_all_to_data()
        for p in self.panes:
            p.set_camera(self.shared_cam)
        self.update_frame()
        
    def _apply_view_mode(self, cam, mode):
        # Camera is entirely determined by current mode + bbox
        self._fit_all_to_data()

    # Keyboard navigation
    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Right:
            self.next_frame()
            return
        elif e.key() == QtCore.Qt.Key_Left:
            self.prev_frame()
            return
        elif e.key() == QtCore.Qt.Key_R:
            # exactly the same path as the button, now that keyboard focus is ours
            self.reset_view()
            return
        elif e.key() == QtCore.Qt.Key_Escape:
            self.close()
            return
        else:
            super().keyPressEvent(e)

    def on_size_changed(self, v: int):
        # int slider 1..120  ->  0.1..12.0 px
        self.point_size = v / 10.0
        # show "2.0" etc.
        if hasattr(self, "lbl_size_val") and self.lbl_size_val is not None:
            self.lbl_size_val.setText(f"{self.point_size:.1f}")
        # apply to all panes
        for p in self.panes:
            p.set_point_size(self.point_size)

    def prev_frame(self):
        if self.num_frames == 0:
            return
        self.idx = (self.idx - 1) % self.num_frames
        self.update_frame()

    def next_frame(self):
        if self.num_frames == 0:
            return
        self.idx = (self.idx + 1) % self.num_frames
        self.update_frame()

    def update_frame(self):        
        if self.num_frames == 0 or not self.file_lists:
            self.lbl_idx.setText("Frame 0 / 0")
            self.lbl_name.setText("No folders selected")
            return

        # Choose name from first pair’s file basename (aligned)
        ref_name = os.path.basename(self.file_lists[0][self.idx])
        self.lbl_idx.setText(self._idx_text())
        self.lbl_name.setText(ref_name)

        # 1) Load all clouds first (no rendering, no camera fit)
        batch = []
        for v_idx in range(len(self.panes)):
            path = self.file_lists[v_idx][self.idx]
            pts, cols = load_cloud(path)
            batch.append((pts, cols))

        # 2) Now update all panes only AFTER all bounding boxes are known
        for v_idx, pane in enumerate(self.panes):
            pane.set_title(os.path.basename(self.folders[v_idx]))
            pts, cols = batch[v_idx]
            pane.set_points(pts, cols, self.point_size)
            pane.set_point_size(self.point_size)
            
        self._fit_all_to_data()
        
        # Render all panes ONCE (this prevents shrink-then-grow)
        for p in self.panes:
            p.plotter.render()
    
        self._save_state()
        
    def closeEvent(self, event):
        for p in self.panes:
            p.close()  # gracefully destroy all VTK windows
        event.accept()


# ────────── CLI ──────────

def build_argparser():
    ap = argparse.ArgumentParser(description="Fast PLY/PCD viewer (1–4 folders). Use Left/Right to navigate.")
    ap.add_argument("--folders", nargs="+", default=DEFAULT_FOLDERS,
                    help="One to four folders (defaults = Paper 2b paths). "
                         "Subfolders are crawled; files paired by basename intersection.")
    return ap

def main():
    parser = build_argparser()
    args = parser.parse_args()
    folders = args.folders
    # app.use_app('pyqt5')
    qapp = QtWidgets.QApplication(sys.argv)
    if os.path.exists(APP_ICON):
        qapp.setWindowIcon(QtGui.QIcon(APP_ICON))
    win = MainWindow(folders)
    win.showMaximized()                                      # maximized
    win.raise_()
    win.activateWindow()                                     # focus to front
    win.show()
    sys.exit(qapp.exec_())

if __name__ == "__main__":
    main()