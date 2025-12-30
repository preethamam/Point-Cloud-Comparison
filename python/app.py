#!/usr/bin/env python3
# Fast 2×2 (or 1–4 panes) point cloud viewer — PyQt5 + VisPy
# • Recursively loads .ply/.pcd from subfolders
# • Pairs (0↔1) and (2↔3) by basename intersection, lexicographically
# • Left/Right arrow + buttons for navigation
# • Point size slider (fast, continuous)
# • Titles as QLabels (no clipping)
# • Linked orthographic cameras, top-down XY view

import argparse
import glob
import hashlib
import json
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import pyvista as pv
from appdirs import user_data_dir
from joblib import Parallel, delayed
from natsort import natsorted
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor

# ────────── DEFAULT FOLDERS (yours) ──────────
DEFAULT_FOLDERS = []
# [
#     r"F:\Datasets\SyntheticCRACK\Paper 2b - A graph-based 3D point clouds (Blend Comparision)\Colored-AlphaBlend",
#     r"F:\Datasets\SyntheticCRACK\Paper 2b - A graph-based 3D point clouds (Blend Comparision)\Groundtruth-AlphaBlend",
#     r"F:\Datasets\SyntheticCRACK\Paper 2b - A graph-based 3D point clouds (Blend Comparision)\Colored-Shading",
#     r"F:\Datasets\SyntheticCRACK\Paper 2b - A graph-based 3D point clouds (Blend Comparision)\Groundtruth-Shading",
# ]

# ────────── Constants ──────────
POINT_SIZE = 3.8  # initial point size in pixels
TIME_PAUSE = 1.5  # seconds between frames during Loop
APP_ICON = "app.ico"  # place app.ico next to this script or give an absolute path
STATE_JSON = "viewer_state.json"

# create (and ensure) the app-specific folder under user AppData
APP_NAME = "Point Cloud Compare"
state_dir = Path(user_data_dir(APP_NAME, appauthor=False))
state_dir.mkdir(parents=True, exist_ok=True)
STATE_JSON = os.path.join(state_dir, STATE_JSON)

THUMB_MAX_PTS = 150_000   # reviewer-style cap
THUMB_SIZE = 96
NAV_WIDTH = 148  # you can tune this
NAV_NAME_MAX = 20   # adjust to taste
THUMB_DIR = os.path.join(state_dir, "thumbs")
os.makedirs(THUMB_DIR, exist_ok=True)
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

def align_all_by_basename(folder_paths: List[str]) -> Tuple[List[List[str]], int]:
    """
    Align N folders by global basename intersection (case-insensitive).
    Guarantees that index i refers to the SAME filename in all panes.
    """
    all_lists = [list_pc_files(f) for f in folder_paths]
    if any(len(L) == 0 for L in all_lists):
        return [], 0

    maps = [map_by_basename(L) for L in all_lists]
    common = set(maps[0].keys())
    for m in maps[1:]:
        common &= set(m.keys())

    common = natsorted(common)
    if not common:
        return [], 0

    aligned = [[m[k] for k in common] for m in maps]
    return aligned, len(common)

def align_file_sets(folder_paths: List[str]) -> Tuple[List[List[str]], int]:
    """
    1 folder  -> simple listing
    2–4 folders -> globally aligned by basename intersection
    """
    k = len(folder_paths)
    if k == 0:
        raise RuntimeError("No folders provided.")

    if k == 1:
        L0 = list_pc_files(folder_paths[0])
        return [L0], len(L0)

    # 🔥 FIX: global alignment across ALL folders
    aligned, n = align_all_by_basename(folder_paths[:4])
    return aligned, n

def _thumb_key(path: str) -> str:
    st = os.stat(path)
    h = hashlib.md5(
        f"{path}|{st.st_mtime_ns}|{st.st_size}".encode()
    ).hexdigest()
    return h + ".png"

def _thumb_path_for(pc_path: str) -> str:
    return os.path.join(THUMB_DIR, _thumb_key(pc_path))


def generate_thumbnail(pc_path: str, out_png: str, size_px: int = THUMB_SIZE):
    try:
        pc = o3d.io.read_point_cloud(pc_path)
        if pc.is_empty():
            return False

        xyz = np.asarray(pc.points)
        rgb = np.asarray(pc.colors) if pc.has_colors() else None

        if xyz.shape[0] > THUMB_MAX_PTS:
            idx = np.random.choice(xyz.shape[0], THUMB_MAX_PTS, replace=False)
            xyz = xyz[idx]
            if rgb is not None:
                rgb = rgb[idx]

        if rgb is None or rgb.shape[0] != xyz.shape[0]:
            rgb = np.ones((xyz.shape[0], 3), np.float32)

        if rgb.max() > 1.0:
            rgb = np.clip(rgb / 255.0, 0, 1)

        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        dx = max(xmax - xmin, 1e-12)
        dy = max(ymax - ymin, 1e-12)

        W = H = size_px
        ix = ((x - xmin) / dx * (W - 1)).astype(np.int32)
        iy = ((y - ymin) / dy * (H - 1)).astype(np.int32)

        img = np.full((H, W, 3), 255, np.uint8)

        order = np.argsort(z)
        img[H - 1 - iy[order], ix[order]] = (rgb[order] * 255).astype(np.uint8)

        Image.fromarray(img).save(out_png)
        return True
    except Exception:
        return False

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

class NavPane(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(220)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)

        # search
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search / filter (name)")
        self.search.setClearButtonEnabled(True)
        v.addWidget(self.search)

        # list
        self.list = QtWidgets.QListWidget()
        self.list.setViewMode(QtWidgets.QListView.IconMode)
        self.list.setResizeMode(QtWidgets.QListView.Adjust)
        self.list.setWrapping(True)
        self.list.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollPerPixel
        )


        self.list.setIconSize(QtCore.QSize(THUMB_SIZE, THUMB_SIZE))
        self.list.setGridSize(QtCore.QSize(THUMB_SIZE + 24, THUMB_SIZE + 40))
        self.list.setMovement(QtWidgets.QListView.Static)
        self.list.setSpacing(6)
        self.list.setWordWrap(True)
        self.list.setTextElideMode(QtCore.Qt.ElideMiddle)
        self.list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        v.addWidget(self.list, 1)        

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


class FolderGroupDialog(QtWidgets.QDialog):
    """Two-column, row-wise paired folder selector."""

    def __init__(self, parent=None, initial_pairs=None):
        super().__init__(parent)
        self.setWindowTitle("Folders")
        self.resize(820, 420)

        initial_pairs = initial_pairs or []

        root = QtWidgets.QVBoxLayout(self)

        # ---- header ----
        header = QtWidgets.QHBoxLayout()
        h1 = QtWidgets.QLabel("Original Folders")
        h2 = QtWidgets.QLabel("Annotation Folders")
        for h in (h1, h2):
            h.setStyleSheet("font-weight: bold")
            h.setAlignment(QtCore.Qt.AlignCenter)
        header.addWidget(h1)
        header.addWidget(h2)
        root.addLayout(header)

        # ---- lists ----
        lists = QtWidgets.QHBoxLayout()
        self.list_orig = QtWidgets.QListWidget()
        self.list_anno = QtWidgets.QListWidget()
        lists.addWidget(self.list_orig)
        lists.addWidget(self.list_anno)
        root.addLayout(lists, 1)

        # preload
        for o, a in initial_pairs:
            self.list_orig.addItem(o)
            self.list_anno.addItem(a)

        # ---- buttons ----
        btns = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add Pair")
        self.btn_remove = QtWidgets.QPushButton("Remove Pair")
        self.btn_up = QtWidgets.QPushButton("↑")
        self.btn_down = QtWidgets.QPushButton("↓")
        for b in (self.btn_add, self.btn_remove, self.btn_up, self.btn_down):
            btns.addWidget(b)
        btns.addStretch(1)
        root.addLayout(btns)

        # ---- OK / Cancel ----
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        ok = QtWidgets.QPushButton("Save")
        cancel = QtWidgets.QPushButton("Cancel")
        footer.addWidget(ok)
        footer.addWidget(cancel)
        root.addLayout(footer)

        # signals
        self.btn_add.clicked.connect(self.add_pair)
        self.btn_remove.clicked.connect(self.remove_pair)
        self.btn_up.clicked.connect(lambda: self.move_pair(-1))
        self.btn_down.clicked.connect(lambda: self.move_pair(+1))
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)

        # sync selection
        self.list_orig.currentRowChanged.connect(
            self.list_anno.setCurrentRow
        )
        self.list_anno.currentRowChanged.connect(
            self.list_orig.setCurrentRow
        )

    # ---------- actions ----------
    def add_pair(self):
        o = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Original folder"
        )
        if not o:
            return

        a = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Annotation folder"
        )
        if not a:
            return

        self.list_orig.addItem(os.path.abspath(o))
        self.list_anno.addItem(os.path.abspath(a))

    def remove_pair(self):
        r = self.list_orig.currentRow()
        if r >= 0:
            self.list_orig.takeItem(r)
            self.list_anno.takeItem(r)

    def move_pair(self, delta):
        r = self.list_orig.currentRow()
        if r < 0:
            return
        nr = r + delta
        if not (0 <= nr < self.list_orig.count()):
            return

        o = self.list_orig.takeItem(r)
        a = self.list_anno.takeItem(r)
        self.list_orig.insertItem(nr, o)
        self.list_anno.insertItem(nr, a)
        self.list_orig.setCurrentRow(nr)

    def pairs(self):
        return [
            (self.list_orig.item(i).text(),
             self.list_anno.item(i).text())
            for i in range(self.list_orig.count())
        ]

# ────────── Main window ──────────
class MainWindow(QtWidgets.QWidget):
    def __init__(self, folder_paths: List[str]):
        super().__init__()
        if os.path.exists(APP_ICON):
            self.setWindowIcon(QtGui.QIcon(APP_ICON))
        self.setWindowTitle("Point Cloud Compare")
        self.folders = [os.path.abspath(f) for f in folder_paths[:4]]
        self.idx = 0
        self._seen = set()

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
        self._thumbs_generating = False    
        self._thumb_out_by_row = {}
        self._thumb_sweep_timer = None
        
        self.thumb_timer = QtCore.QTimer(self)
        self.thumb_timer.setInterval(200)  # ms
        self.thumb_timer.timeout.connect(self._refresh_nav_icons)

        # ── Top bar: [left zone] … center zone … [right zone]
        root = QtWidgets.QVBoxLayout(self)
                
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(self.splitter, 1)
        
        self.status = QtWidgets.QStatusBar()
        self.status.setSizeGripEnabled(False)
        root.addWidget(self.status)

        # --- nav pane ---
        self.nav = NavPane(self)
        self.nav.setMinimumWidth(NAV_WIDTH)
        self.nav.setMaximumWidth(NAV_WIDTH)                            
        self.splitter.addWidget(self.nav)
        self.splitter.setChildrenCollapsible(False)

        # --- main content ---
        self.main_widget = QtWidgets.QWidget()
        self.splitter.addWidget(self.main_widget)

        root = QtWidgets.QVBoxLayout(self.main_widget)

        # use a 3-column grid so the center column is truly centered
        top = QtWidgets.QGridLayout()
        root.addLayout(top)

        # LEFT (frame + filename)
        left = QtWidgets.QHBoxLayout()
        
        self.btn_nav = QtWidgets.QPushButton("☰")
        self.btn_nav.setFixedWidth(28)
        self.btn_nav.setToolTip("Toggle navigation (N)")
        self.btn_nav.setCheckable(True)
        self.btn_nav.setChecked(True)
        self.btn_nav.setFocusPolicy(QtCore.Qt.NoFocus)
        left.insertWidget(1, self.btn_nav)
        left.addSpacing(0)

        # NEW: Folders… button (kept lightweight; selects 4 folders)
        self.btn_folders = QtWidgets.QPushButton("Folders…")
        self.btn_folders.setFixedWidth(84)
        self.btn_folders.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        left.addSpacing(0)
        left.addWidget(self.btn_folders)

        self.lbl_idx = QtWidgets.QLabel(self._idx_text())
        self.lbl_name = QtWidgets.QLabel("") 

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
    
        # NEW: jump-to-index textbox
        self.edit_jump = QtWidgets.QLineEdit()
        self.edit_jump.setPlaceholderText("Input PC Index and Press Enter")
        self.edit_jump.setFixedWidth(158)
        self.edit_jump.setClearButtonEnabled(True)
        self.edit_jump.setFocusPolicy(QtCore.Qt.ClickFocus)

        # --- center group: view + reset + loop + seconds + jump in box---
        left.addWidget(self.view_combo)
        left.addWidget(self.btn_reset)
        left.addWidget(self.btn_loop)
        left.addWidget(self.spin_pause)
        left.addWidget(self.edit_jump)

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
        self.btn_nav.clicked.connect(self.toggle_nav)
        
        self.btn_del_thumbs = QtWidgets.QPushButton("Delete Thumbnails")
        self.btn_del_thumbs.setToolTip("Delete cached thumbnails")
        self.btn_del_thumbs.setFixedWidth(100)
        self.btn_del_thumbs.setFocusPolicy(QtCore.Qt.NoFocus)

        for b in (self.btn_prev, self.btn_next):
            b.setFixedWidth(86)
            b.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        right.addWidget(self.btn_prev)
        right.addWidget(self.btn_next)
        right.addWidget(self.btn_del_thumbs)


        # --- place into the grid (center stays geometrically centered)
        top.addLayout(left,   0, 0, alignment=QtCore.Qt.AlignLeft)
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
        
        # ── Connections & shortcuts ──        
        self.nav.list.itemClicked.connect(
            lambda it: self._jump_from_nav(it)
        )
        self.nav.list.currentRowChanged.connect(self._jump_from_nav_row)
        
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
        
        QtWidgets.QShortcut(
            QtGui.QKeySequence("Ctrl+Q"),
            self,
            activated=self.close
        )
        
        sc_nav = QtWidgets.QShortcut(QtGui.QKeySequence("N"), self)
        sc_nav.setContext(QtCore.Qt.ApplicationShortcut)
        sc_nav.activated.connect(self._toggle_nav_from_shortcut)

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

        self.nav.search.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.nav.search.textChanged.connect(self._filter_nav_items)
        self.nav.search.installEventFilter(self)
        self.nav.list.installEventFilter(self)

        # prevent arrow keys from landing on controls
        for w in (self.btn_prev, self.btn_next, self.btn_reset, self.view_combo, self.btn_loop):
            w.setFocusPolicy(QtCore.Qt.NoFocus)

        # ── Signals
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_next.clicked.connect(self.next_frame)
        self.btn_del_thumbs.clicked.connect(self.delete_thumbnails)

        self.slider.valueChanged.connect(self.on_size_changed)
        self.view_combo.currentIndexChanged.connect(self.on_view_mode_changed)
        self.btn_reset.clicked.connect(self.reset_view)

    
        # NEW: Loop timer + wiring
        self.loop_timer = QtCore.QTimer(self)
        self.loop_timer.setInterval(800)
        self.loop_timer.timeout.connect(self._advance_or_stop)
        self.btn_loop.toggled.connect(self.toggle_loop)
        self.edit_jump.returnPressed.connect(self._jump_to_index)

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

    def _jump_from_nav_row(self, row: int):
        # Called when user moves selection with keyboard (Up/Down)
        if row < 0 or row == self.idx:
            return

        self.idx = row
        self.update_frame()

    def showEvent(self, event):
        super().showEvent(event)

        # Ensure splitter respects fixed nav width once geometry is known
        if self.splitter.count() >= 2:
            total = self.splitter.width()
            self.splitter.setSizes([NAV_WIDTH, max(1, total - NAV_WIDTH)])

    def _filter_nav_items(self, text: str):
        """
        Filter LEFT navigation items by:
        - index (e.g. '12', '0012')
        - filename
        """
        text = text.strip().lower()

        if not self.file_lists:
            return

        files = self.file_lists[0]

        for row in range(self.nav.list.count()):
            item = self.nav.list.item(row)
            path = files[row]

            # index match (reviewer-style)
            idx_str = f"{row+1:04d}"

            # filename match (full name, not truncated)
            fname = os.path.basename(path).lower()

            visible = (
                not text or
                text in idx_str or
                text in fname
            )

            item.setHidden(not visible)

        # ensure current frame remains visible
        if 0 <= self.idx < self.nav.list.count():
            cur_item = self.nav.list.item(self.idx)
            if cur_item and not cur_item.isHidden():
                self.nav.list.scrollToItem(cur_item)

    def _toggle_nav_from_shortcut(self):
        # Toggle the button state explicitly
        self.btn_nav.setChecked(not self.btn_nav.isChecked())
        self.toggle_nav()

    def toggle_nav(self):
        if self.nav.isVisible():
            self._nav_last_width = self.nav.width()
            self.nav.hide()
        else:
            self.nav.show()
            w = getattr(self, "_nav_last_width", NAV_WIDTH)
            self.nav.setFixedWidth(w)
            
        # 🔥 critical: refit after splitter geometry change
        QtCore.QTimer.singleShot(0, self._refit_after_resize)
    
    def _sync_nav_selection(self):
        """Ensure LEFT nav selection follows current index."""
        if not self.nav.list.count():
            return

        row = self.idx
        if 0 <= row < self.nav.list.count():
            self.nav.list.setCurrentRow(row)
            self.nav.list.scrollToItem(
                self.nav.list.item(row),
                QtWidgets.QAbstractItemView.PositionAtCenter
            )

    def populate_nav(self):
        self.nav.list.clear()
        self._seen = set()

        if not self.file_lists:
            return

        files = self.file_lists[0]  # reference list

        for i, p in enumerate(files):
            name_full = os.path.basename(p)
            name_disp = (
                name_full[:NAV_NAME_MAX - 1] + "…"
                if len(name_full) > NAV_NAME_MAX
                else name_full
            )

            idx_txt = f"{i+1:04d}"

            item = QtWidgets.QListWidgetItem(f"{idx_txt}\n{name_disp}")
            item.setToolTip(name_full)

            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
            item.setData(QtCore.Qt.UserRole, i)
            self.nav.list.addItem(item)
       
        # --- enqueue thumbnails (reviewer-style realtime) ---
        self._thumb_out_by_row.clear()
        jobs = []

        files = self.file_lists[0]

        for row, p in enumerate(files):
            out_png = _thumb_path_for(p)
            self._thumb_out_by_row[row] = out_png

            if not os.path.exists(out_png):
                jobs.append((p, out_png))

        if jobs and not self._thumbs_generating:
            self._thumbs_generating = True

            def worker(j=jobs):
                try:
                    Parallel(
                        n_jobs=-1,
                        backend="threading",   # GUI-safe
                        verbose=0
                    )(
                        delayed(generate_thumbnail)(p, out)
                        for p, out in j
                    )
                finally:
                    self._thumbs_generating = False

            threading.Thread(target=worker, daemon=True).start()

        # start sweeper
        if self._thumb_sweep_timer is None:
            self._thumb_sweep_timer = QtCore.QTimer(self)
            self._thumb_sweep_timer.timeout.connect(self._sweep_thumb_icons)

        self._thumb_sweep_timer.start(120)
        QtCore.QTimer.singleShot(0, self._sync_nav_selection)

    def _sweep_thumb_icons(self):
        if not self._thumb_out_by_row:
            self._thumb_sweep_timer.stop()
            return

        rows = list(self._thumb_out_by_row.keys())
        batch = 80
        updated = False

        for _ in range(batch):
            if not rows:
                break

            r = rows[0]
            out_png = self._thumb_out_by_row.get(r)

            if out_png and os.path.exists(out_png):
                it = self.nav.list.item(r)
                if it is not None:
                    it.setIcon(QtGui.QIcon(out_png))
                    updated = True

                self._thumb_out_by_row.pop(r, None)
                rows.pop(0)
            else:
                rows.append(rows.pop(0))

        if not self._thumb_out_by_row:
            self._thumb_sweep_timer.stop()

        if updated:
            self.nav.list.viewport().update()      
        
    def _refresh_nav_icons(self):
        all_done = True

        for i in range(self.nav.list.count()):
            it = self.nav.list.item(i)
            p = self.file_lists[0][i]
            tp = _thumb_path_for(p)

            if os.path.exists(tp):
                if it.icon().isNull():
                    it.setIcon(QtGui.QIcon(tp))
            else:
                all_done = False

        if all_done:
            self.thumb_timer.stop()

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:

            # ⬅️⬅️ FIX: arrow keys while nav list has focus
            if obj is self.nav.list:
                if event.key() == QtCore.Qt.Key_Right:
                    self.next_frame()
                    return True
                elif event.key() == QtCore.Qt.Key_Left:
                    self.prev_frame()
                    return True

            fw = QtWidgets.QApplication.focusWidget()

            if fw is self.nav.search:
                if event.key() == QtCore.Qt.Key_Left:
                    self.prev_frame()
                    return True
                elif event.key() == QtCore.Qt.Key_Right:
                    self.next_frame()
                    return True
                elif event.key() == QtCore.Qt.Key_Escape:
                    self.nav.search.clear()
                    self.setFocus(QtCore.Qt.OtherFocusReason)
                    return True

        if event.type() == QtCore.QEvent.Wheel:
            self._on_wheel(event, obj)
            return True

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
        # rebuild existing pairs from flat self.folders
        pairs = []
        f = self.folders
        for i in range(0, len(f), 2):
            if i + 1 < len(f):
                pairs.append((f[i], f[i + 1]))

        dlg = FolderGroupDialog(self, initial_pairs=pairs)
        if dlg.exec_():
            flat = []
            for o, a in dlg.pairs():
                flat.extend([o, a])
            if flat:
                self.apply_new_folders(flat)


    def apply_new_folders(self, folders):
        self.folders = [os.path.abspath(f) for f in folders[:4]]
        self.file_lists, self.num_frames = align_file_sets(self.folders)

        if self.num_frames > 0:
            self.populate_nav()
            
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

    def _jump_to_index(self):
        if self.num_frames == 0:
            return

        txt = self.edit_jump.text().strip()
        if not txt.isdigit():
            return

        idx = int(txt) - 1   # user sees 1-based indexing
        if 0 <= idx < self.num_frames:
            self.idx = idx
            self.update_frame()

        self.edit_jump.clear()
        self.setFocus(QtCore.Qt.OtherFocusReason)  # ← restore arrow-key nav

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

    def _lock_clipping_range(self):
        cam = self.shared_cam
        fp = np.array(cam.GetFocalPoint())
        pos = np.array(cam.GetPosition())
        dist = np.linalg.norm(pos - fp)

        # Conservative, stable clipping range
        near = max(dist * 0.001, 1e-6)
        far  = dist * 10.0

        cam.SetClippingRange(near, far)

    def _ortho_scale_from_bbox(self, cam, mn, mx, aspect, pad=1.03):
        """
        Compute ParallelScale for an orthographic camera by projecting
        the 3D bounding box onto the camera view plane.
        """
        # 8 corners of bounding box
        corners = np.array([
            [mn[0], mn[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]],
            [mx[0], mx[1], mx[2]],
        ], dtype=float)

        # Camera basis
        pos = np.array(cam.GetPosition())
        fp  = np.array(cam.GetFocalPoint())
        up  = np.array(cam.GetViewUp())
        view_dir = fp - pos
        view_dir /= np.linalg.norm(view_dir)

        right = np.cross(view_dir, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, view_dir)

        # Project corners into camera plane
        rel = corners - fp
        x = rel @ right
        y = rel @ up

        w = x.max() - x.min()
        h = y.max() - y.min()

        return 0.5 * max(h, w / aspect) * pad

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # Defer refit until Qt layout settles (critical)
        QtCore.QTimer.singleShot(0, self._refit_after_resize)
        
    def _refit_after_resize(self):
        if self.num_frames == 0 or not self.panes:
            return

        self._fit_all_to_data()

        for p in self.panes:
            p.plotter.render()

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

        cx, cy, cz = (mn + mx) * 0.5
        dx, dy, dz = (mx - mn)
        
        # Defensive defaults (prevents UnboundLocalError)
        plane_w = float(dx)
        plane_h = float(dy)

        # Comfortable camera distance (still fine to use 3D diagonal)
        diag = float(np.linalg.norm([dx, dy, dz]) or 1.0)
        dist = diag * 1.3

        cam = self.shared_cam
        mode = self.view_combo.currentText()
        mode = mode.split()[0]   # "SW Isometric view (...)" → "SW"

        cam.SetFocalPoint(cx, cy, cz)

        # --- determine view direction + view-plane extents (for tight ortho fit) ---
        if mode.startswith("Top"):
            cam.SetPosition(cx, cy, cz + dist)
            cam.SetViewUp(0, 1, 0)
            plane_w, plane_h = float(dx), float(dy)

        elif mode.startswith("Bottom"):
            cam.SetPosition(cx, cy, cz - dist)
            cam.SetViewUp(0, 1, 0)
            plane_w, plane_h = float(dx), float(dy)

        elif mode.startswith("Front"):
            cam.SetPosition(cx, cy - dist, cz)
            cam.SetViewUp(0, 0, 1)
            plane_w, plane_h = float(dx), float(dz)

        elif mode.startswith("Back"):
            cam.SetPosition(cx, cy + dist, cz)
            cam.SetViewUp(0, 0, 1)
            plane_w, plane_h = float(dx), float(dz)

        elif mode.startswith("Left"):
            cam.SetPosition(cx - dist, cy, cz)
            cam.SetViewUp(0, 0, 1)
            plane_w, plane_h = float(dy), float(dz)

        elif mode.startswith("Right"):
            cam.SetPosition(cx + dist, cy, cz)
            cam.SetViewUp(0, 0, 1)
            plane_w, plane_h = float(dy), float(dz)

        else:
            # --- Isometric views: exact orthographic fit via bbox projection ---
            if mode.startswith("SW"):
                cam.SetPosition(cx - dist, cy - dist, cz + dist)
            elif mode.startswith("SE"):
                cam.SetPosition(cx + dist, cy - dist, cz + dist)
            elif mode.startswith("NW"):
                cam.SetPosition(cx - dist, cy + dist, cz + dist)
            elif mode.startswith("NE"):
                cam.SetPosition(cx + dist, cy + dist, cz + dist)

            cam.SetViewUp(0, 0, 1)
            cam.SetParallelProjection(True)

            # viewport aspect
            aspect = 1.0
            if self.panes:
                wpx, hpx = self.panes[0].plotter.window_size
                if hpx:
                    aspect = wpx / hpx

            scale = self._ortho_scale_from_bbox(cam, mn, mx, aspect, pad=1.03)
            cam.SetParallelScale(scale)

        # --- compute orthographic scale using viewport aspect ratio ---
        aspect = 1.0
        if self.panes:
            wpx, hpx = self.panes[0].plotter.window_size
            if hpx:
                aspect = float(wpx) / float(hpx)

        plane_w = max(plane_w, 1e-12)
        plane_h = max(plane_h, 1e-12)

        # ParallelScale is half the visible world-height; width constraint uses aspect
        pad = 1.03  # small padding; tune 1.03–1.10
        half_h_needed = 0.5 * max(plane_h, plane_w / max(aspect, 1e-6))
        cam.SetParallelScale(half_h_needed * pad)

        cam.Modified()

        # IMPORTANT: don't render here (prevents “splash”)
        self._lock_clipping_range()


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
            # ESC = soft cancel only
            self.nav.search.clear()
            self.setFocus(QtCore.Qt.OtherFocusReason)
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
        self._seen.add(self.idx)
        self.idx = (self.idx - 1) % self.num_frames
        self.update_frame()

    def next_frame(self):
        if self.num_frames == 0:
            return
        self._seen.add(self.idx)
        self.idx = (self.idx + 1) % self.num_frames
        self.update_frame()

    def update_frame(self):
        if self.num_frames == 0 or not self.file_lists:
            self.lbl_idx.setText("Frame 0 / 0")
            self.lbl_name.setText("No folders selected")
            return

        # Freeze Qt repaints for the VTK widgets (prevents splash/flicker)
        for p in self.panes:
            try:
                p.plotter.interactor.setUpdatesEnabled(False)
            except Exception:
                pass

        try:
            ref_name = os.path.basename(self.file_lists[0][self.idx])
            self.lbl_idx.setText(self._idx_text())
            self.lbl_name.setText(ref_name)

            self.setWindowTitle(f"{APP_NAME} — {ref_name}   ({self.idx + 1}/{self.num_frames})")
            self.status.showMessage(f"Viewing: {ref_name}")

            # 1) Load all clouds first
            batch = []
            for v_idx in range(len(self.panes)):
                path = self.file_lists[v_idx][self.idx]
                pts, cols = load_cloud(path)
                batch.append((pts, cols))

            # 2) Update all panes (no renders yet)
            for v_idx, pane in enumerate(self.panes):
                pane.set_title(os.path.basename(self.folders[v_idx]))
                pts, cols = batch[v_idx]
                pane.set_points(pts, cols, self.point_size)

            # 3) Fit camera once (no renders inside)
            self._fit_all_to_data()

            # 4) Render once per pane (after everything is consistent)
            for p in self.panes:
                p.plotter.render()

            self._seen.add(self.idx)
            self._refresh_nav_styles()
            self._save_state()
            self._sync_nav_selection()

        finally:
            for p in self.panes:
                try:
                    p.plotter.interactor.setUpdatesEnabled(True)
                except Exception:
                    pass
    
    def _refresh_nav_styles(self):
        for i in range(self.nav.list.count()):
            it = self.nav.list.item(i)
            if i in self._seen:
                it.setBackground(QtGui.QColor("#d0e7ff"))
            else:
                it.setBackground(QtGui.QColor("white"))
                
    def _jump_from_nav(self, item):
        idx = item.data(QtCore.Qt.UserRole)
        if idx is not None:
            self.idx = idx
            self.update_frame()
            self._sync_nav_selection()
        
    def closeEvent(self, event):
        for p in self.panes:
            p.close()  # gracefully destroy all VTK windows
        event.accept()
        
    def delete_thumbnails(self):
        try:
            if os.path.isdir(THUMB_DIR):
                import shutil
                shutil.rmtree(THUMB_DIR)
                QtWidgets.QMessageBox.information(
                    self, "Thumbnails Deleted",
                    "Thumbnail cache has been deleted.\nRestart or reopen navigation to regenerate."
                )
            else:
                QtWidgets.QMessageBox.information(
                    self, "No Thumbnails",
                    "No thumbnail cache folder found."
                )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Error",
                f"Failed to delete thumbnails:\n{e}"
            )



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
    
    # ⏱️ defer thumbnails until UI is alive
    QtCore.QTimer.singleShot(0, win.populate_nav)

    win.raise_()
    win.activateWindow()                                     # focus to front
    win.show()
    sys.exit(qapp.exec_())

if __name__ == "__main__":
    main()