"""
app.py — Desktop GUI for the ESRG pipeline.

Purpose : Load an MRI slice, run the pipeline, inspect any stage, and read
          evaluation scores in a flat, light, clinical-workstation interface.
Function : CustomTkinter app in a header / 3-column / footer grid. Left: input,
          method, seeding, ablation and hyperparameter controls. Centre: stage
          tabs over two equal panes (Original | Segmented Output) with zoom / pan
          controls. Right: collapsible stage telemetry: a seed table and a Details
          view that explains the selected stage end to end (inputs, equation,
          computation, outputs; esrg/stage_report.py). Footer: one
          line of evaluation metrics. The pipeline runs off the UI thread.
Notes   : Strictly flat: no gradients, bevels or shadows; every widget is square
          (corner_radius=0) apart from the radio buttons' circular indicators.
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esrg import Config, run
from esrg import explain as evalx
from esrg import pixel_report
from esrg import stage_report
from esrg import preprocessing as pre
from esrg import visualize as viz
from esrg.io_utils import find_mask_for, load_image

# ── Clinical Flat Theme Tokens ───────────────────────────────────────────────
WHITE       = "#FFFFFF"   # header, centre column, footer, telemetry
PANEL       = "#F4F5F7"   # control panels
BORDER      = "#DFE1E6"   # 1px rules and idle-control outlines
BORDER_DARK = "#B3BAC5"   # checkbox / radio rings, scrollbar thumbs
OBJ_IDLE    = "#E4E6EA"   # header OBJ buttons at rest
VIEWPORT    = "#000000"   # MRI panes

TEXT        = "#172B4D"
MUTED       = "#5E6C84"
FAINT       = "#97A0AF"
ON_BLACK    = "#9AA3B2"   # hint text drawn on the black panes

BLUE        = "#0066CC"
BLUE_HOV    = "#0052A3"

OK_CLR      = "#1E8E3E"
IN_BG       = "#E8F0FB"   # telemetry INPUT band
OUT_BG      = "#E6F4EA"   # telemetry OUTPUT band
WARN_CLR    = "#B26A00"
FAIL_CLR    = "#C5221F"

# Manual seed regions share the tessellation palette (esrg.visualize.SEED_COLORS):
# region 1 is whichever region the user treats as the structure of interest,
# region 2+ are the other competing regions. SRG itself does not distinguish them.
SEED_RGB = viz.SEED_COLORS

DIAG_PANEL_W = 380    # telemetry panel width in Seeds mode (px)
DETAIL_PANEL_W = 520  # wider in Details mode, where stage tables are shown
RAIL_W       = 34   # width of the strip left behind when telemetry is collapsed
LEFT_W       = 232  # default width of the control panel; drag its right edge to resize
LEFT_MIN     = 218  # narrowest that still fits the longest checkbox label
LEFT_MAX     = 420
CENTER_MIN   = 420  # the viewer never gets squeezed below this by the splitter

ZOOM_MIN = 0.25
ZOOM_MAX = 8.0

# Header OBJ buttons jump to the stage that shows each thesis objective's output.
OBJ_STAGE = {1: "seed", 2: "log", 3: "growth"}

# Footer metrics: (label, key in Result.scores, format)
FOOTER_METRICS = (
    ("Dice Coefficient", "dsc", "{:.3f}"),
    ("Jaccard Index", "iou", "{:.3f}"),
    ("Precision", "precision", "{:.3f}"),
    ("Recall", "recall", "{:.3f}"),
    ("Boundary Error (HD95)", "hd95", "{:.2f} px"),
)

# Typography. CTk widgets take pixel sizes; plain Tk widgets (Text, Treeview) take points.
F_TITLE = ("Segoe UI", 17, "bold")
F_BODY  = ("Segoe UI", 13)
F_BOLD  = ("Segoe UI", 13, "bold")
F_SMALL = ("Segoe UI", 12)
F_CAP   = ("Segoe UI", 11, "bold")
F_RUN   = ("Segoe UI", 14, "bold")
TK_UI   = ("Segoe UI", 9)
TK_BOLD = ("Segoe UI", 9, "bold")
TK_MONO = ("Consolas", 9) if sys.platform == "win32" else ("Menlo", 8)


def _rule(parent, vertical=False):
    """1px flat divider (plain Tk frame so DPI scaling never thickens it)."""
    if vertical:
        return tk.Frame(parent, bg=BORDER, width=1)
    return tk.Frame(parent, bg=BORDER, height=1)


def _hide(w):
    """Unmap a gridded CTk widget without losing its Tk grid state.

    CTk keeps a record of every widget's last geometry call and replays it whenever the window's
    DPI scale changes (e.g. after a resize onto another monitor), which would re-grid anything
    that was merely grid_remove()d. grid_forget() clears that record but also resets the widget's
    own grid_propagate / row and column weights as a container, so remove it and drop the
    record by hand. The next show-site calls .grid(...) again, which re-records it."""
    w.grid_remove()
    w._last_geometry_manager_call = None


def _gray_rgb(a):
    g = np.clip(a, 0, 255).astype(np.uint8)
    return np.stack([g] * 3, axis=-1)


class ESRGApp:
    def __init__(self, root):
        self.root = root
        ctk.set_appearance_mode("light")
        root.title("Enhanced Seeded Region Growing Algorithm in MRI Image Segmentation")
        root.configure(fg_color=WHITE)
        self._fit_window(1440, 880, 1120, 700)

        self._init_ttk_style()

        self.cfg = Config()
        self.image_path = None
        self.result = None
        self.current = "final"
        self.manual_points = []
        self._run_points = set()  # (row, col, type) clicks the last run actually used
        self.raw_image = None  # grayscale float64 preview shown before the pipeline runs
        self.norm_image = None  # pre.normalize(raw_image): what the pipeline grows on
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_start = None
        self._left_press = None
        self._left_dragging = False
        self._photos = {}
        self._pane_cache = (None, None)
        self._render_pending = False
        self.tel_visible = True
        self._left_w = LEFT_W
        self.pan_tool = tk.BooleanVar(value=False)
        # Evaluation step: which metrics the Details view computes and explains.
        self.eval_vars = {k: tk.BooleanVar(value=True) for k, _, _ in evalx.METRICS}
        self.run_cfg = None  # configuration of the last run (supplies λ for leakage)

        self.opacity = tk.DoubleVar(value=0.55)
        self.error_mode = tk.BooleanVar(value=False)
        self.compare_overlay = tk.BooleanVar(value=False)
        self.method = tk.StringVar(value="esrg")
        self.seed_mode = tk.StringVar(value="auto")
        self.seed_type = tk.IntVar(value=1)
        self.use_local = tk.BooleanVar(value=True)
        self.use_stop = tk.BooleanVar(value=True)
        self.use_log = tk.BooleanVar(value=self.cfg.use_log)
        self.purify_manual_seed = tk.BooleanVar(value=self.cfg.purify_manual_seed)
        self.k_local = tk.DoubleVar(value=self.cfg.k_local)
        self.radius = tk.IntVar(value=self.cfg.local_radius)
        self.classes = tk.IntVar(value=self.cfg.otsu_classes)
        self.tel_mode = tk.StringVar(value="Seeds")

        self._build()

    @property
    def _scale(self):
        """Current DPI scale of the window. Read live: it changes when the window is moved or
        resized onto a monitor with a different scaling."""
        return self.root._get_window_scaling()

    def _px(self, n):
        return int(round(n * self._scale))

    def _fit_window(self, w, h, min_w, min_h):
        """Open at (w, h) in CTk's DPI-independent units, shrunk to fit the screen with
        room for the taskbar, and centred horizontally. CTk multiplies the size by the
        display scale but takes the +x+y offset in physical pixels."""
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = min(w, int(sw * 0.94)), min(h, int(sh * 0.88))
        x = max(0, int((sw - w) * self._scale / 2))
        self.root.geometry(f"{w}x{h}+{x}+{self._px(8)}")
        self.root.minsize(min(min_w, w), min(min_h, h))

    def _init_ttk_style(self):
        """Flat Treeview + no bevels: the only ttk widget in the app is the seed table."""
        style = ttk.Style(self.root)
        style.theme_use("clam")
        row_h = tkfont.Font(font=TK_UI).metrics("linespace") + 10
        style.configure("Seeds.Treeview", background=WHITE, fieldbackground=WHITE,
                        foreground=TEXT, rowheight=row_h, borderwidth=0, relief="flat",
                        font=TK_UI)
        style.configure("Seeds.Treeview.Heading", background=PANEL, foreground=MUTED,
                        relief="flat", borderwidth=0, font=TK_BOLD, padding=(6, 5))
        style.map("Seeds.Treeview", background=[("selected", BLUE)],
                  foreground=[("selected", WHITE)])
        style.map("Seeds.Treeview.Heading", background=[("active", BORDER)])
        style.layout("Seeds.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])

    # ── Widget factories ─────────────────────────────────────────────────────
    def _btn(self, parent, text, command, primary=False, width=None, height=30, font=F_BODY):
        """Square flat button: idle = PANEL with a 1px rule, primary = solid blue."""
        kw = dict(text=text, command=command, corner_radius=0, font=font, height=height,
                  border_width=1)
        if width:
            kw["width"] = width
        if primary:
            return ctk.CTkButton(parent, fg_color=BLUE, hover_color=BLUE_HOV, text_color=WHITE,
                                 border_color=BLUE, text_color_disabled=WHITE, **kw)
        return ctk.CTkButton(parent, fg_color=PANEL, hover_color=BORDER, text_color=TEXT,
                             border_color=BORDER, **kw)

    @staticmethod
    def _style_toggle(btn, active):
        """Selected = solid blue; otherwise the idle flat style."""
        if active:
            btn.configure(fg_color=BLUE, hover_color=BLUE, text_color=WHITE, border_color=BLUE)
        else:
            btn.configure(fg_color=PANEL, hover_color=BORDER, text_color=TEXT, border_color=BORDER)

    def _check(self, parent, text, var, command=None):
        return ctk.CTkCheckBox(parent, text=text, variable=var, command=command, font=F_SMALL,
                               text_color=TEXT, corner_radius=0, border_width=1,
                               checkbox_width=16, checkbox_height=16, fg_color=BLUE,
                               hover_color=BLUE_HOV, border_color=BORDER_DARK,
                               checkmark_color=WHITE)

    def _radio(self, parent, text, var, value, command=None):
        return ctk.CTkRadioButton(parent, text=text, variable=var, value=value, command=command,
                                  font=F_SMALL, text_color=TEXT, radiobutton_width=16, radiobutton_height=16,
                                  border_width_unchecked=1, border_width_checked=5,
                                  fg_color=BLUE, hover_color=BLUE_HOV, border_color=BORDER_DARK)

    # ── Master Layout ────────────────────────────────────────────────────────
    def _build(self):
        r = self.root
        r.grid_columnconfigure(0, weight=1)
        r.grid_rowconfigure(2, weight=1)

        self._build_header(r)
        _rule(r).grid(row=1, column=0, sticky="ew")

        body = ctk.CTkFrame(r, fg_color=WHITE, corner_radius=0)
        body.grid(row=2, column=0, sticky="nsew")
        # cols: 0 left | 1 rule | 2 centre (weight 1) | 3 rule | 4 telemetry or rail
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(2, weight=1)

        self.left = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=0, width=LEFT_W)
        self.left.grid(row=0, column=0, sticky="ns")
        self.left.grid_propagate(False)
        self._build_left(self.left)
        self._build_splitter(body)

        center = ctk.CTkFrame(body, fg_color=WHITE, corner_radius=0)
        center.grid(row=0, column=2, sticky="nsew")
        self._build_center(center)
        _rule(body, True).grid(row=0, column=3, sticky="ns")

        self.tel_panel = ctk.CTkFrame(body, fg_color=WHITE, corner_radius=0, width=DIAG_PANEL_W)
        self.tel_panel.grid(row=0, column=4, sticky="ns")
        self.tel_panel.grid_propagate(False)
        self._build_telemetry(self.tel_panel)

        self.rail = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=0, width=RAIL_W)
        self.rail.grid_propagate(False)
        self._btn(self.rail, "|<", self._toggle_telemetry, width=RAIL_W - 8, height=28,
                  font=F_CAP).place(x=4, y=8)

        _rule(r).grid(row=3, column=0, sticky="ew")
        self._build_footer(r)
        self._sync_overlay_toggle()
        self._sync_obj_buttons()
        self._update_metrics()

    # ── Resizable control panel ──────────────────────────────────────────────
    def _build_splitter(self, body):
        """Draggable divider between the control panel and the viewer. It looks like the
        usual 1px rule, but its 5px hit area turns blue on hover. Double-click resets."""
        self._split = tk.Frame(body, bg=WHITE, width=5, cursor="sb_h_double_arrow")
        self._split.grid(row=0, column=1, sticky="ns")
        self._split_line = tk.Frame(self._split, bg=BORDER, width=1)
        self._split_line.place(x=0, y=0, relheight=1)
        self._split_drag = None
        for w in (self._split, self._split_line):
            w.bind("<Enter>", lambda e: self._split_line.configure(bg=BLUE))
            w.bind("<Leave>", lambda e: self._split_drag or self._split_line.configure(bg=BORDER))
            w.bind("<ButtonPress-1>", self._split_press)
            w.bind("<B1-Motion>", self._split_move)
            w.bind("<ButtonRelease-1>", self._split_release)
            w.bind("<Double-Button-1>", lambda e: self._set_left_width(LEFT_W))

    def _split_press(self, event):
        self._split_drag = (event.x_root, self._left_w)
        self._split_line.configure(bg=BLUE)

    def _split_move(self, event):
        if self._split_drag:
            x0, w0 = self._split_drag
            # CTk sizes are DPI-independent units; pointer coordinates are physical pixels.
            self._set_left_width(w0 + (event.x_root - x0) / self._scale)

    def _split_release(self, _event):
        self._split_drag = None
        self._split_line.configure(bg=BORDER)

    def _set_left_width(self, w):
        """Resize the control panel, keeping the viewer and the telemetry column usable."""
        right = (self.tel_panel if self.tel_visible else self.rail).cget("width")
        room = self.root.winfo_width() / self._scale - right - CENTER_MIN
        w = int(round(max(LEFT_MIN, min(LEFT_MAX, room, w))))
        if w == self._left_w:
            return
        self._left_w = w
        self.left.configure(width=w)
        # Wrapped labels follow the panel so they never run under the scrollbar.
        for lbl in (self.file_lbl, self.status_lbl):
            lbl.configure(wraplength=w - 50)

    # ── Header ───────────────────────────────────────────────────────────────
    def _build_header(self, parent):
        bar = ctk.CTkFrame(parent, fg_color=WHITE, corner_radius=0, height=58)
        bar.grid(row=0, column=0, sticky="ew")
        bar.pack_propagate(False)
        ctk.CTkLabel(bar, text="Enhanced Seeded Region Growing Algorithm in MRI Image Segmentation",
                     font=F_TITLE, text_color=TEXT).pack(side="left", padx=20)

        # Packed right-to-left so they read OBJ 1, OBJ 2, OBJ 3 from the left.
        self.obj_buttons = {}
        self._obj_active = None
        for n in (3, 2, 1):
            b = ctk.CTkButton(bar, text=f"OBJ {n}", width=74, height=32, corner_radius=0,
                              font=F_CAP, border_width=0, fg_color=OBJ_IDLE, hover_color=BLUE,
                              text_color=TEXT, text_color_disabled=FAINT,
                              command=lambda n=n: self._on_obj(n))
            b.pack(side="right", padx=(0, 20 if n == 3 else 6))
            b.bind("<Enter>", lambda e, n=n: self._obj_hover(n, True), add="+")
            b.bind("<Leave>", lambda e, n=n: self._obj_hover(n, False), add="+")
            self.obj_buttons[n] = b

    def _obj_enabled(self, n):
        return self.result is not None and self.result.stage(OBJ_STAGE[n]) is not None

    def _obj_hover(self, n, inside):
        # CTkButton animates only the fill on hover; swap the label colour here.
        if not self._obj_enabled(n) or n == self._obj_active:
            return
        self.obj_buttons[n].configure(text_color=WHITE if inside else TEXT)

    def _on_obj(self, n):
        if self._obj_enabled(n):
            self._select(OBJ_STAGE[n])

    def _sync_obj_buttons(self):
        self._obj_active = next(
            (n for n, k in OBJ_STAGE.items() if self._obj_enabled(n) and self.current == k), None)
        for n, b in self.obj_buttons.items():
            if not self._obj_enabled(n):
                b.configure(state="disabled", fg_color=OBJ_IDLE, text_color=FAINT)
            elif n == self._obj_active:
                b.configure(state="normal", fg_color=BLUE, hover_color=BLUE, text_color=WHITE)
            else:
                b.configure(state="normal", fg_color=OBJ_IDLE, hover_color=BLUE, text_color=TEXT)

    # ── Left column: controls ────────────────────────────────────────────────
    def _section_header(self, parent, text):
        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.pack(fill="x", pady=(10, 4))
        ctk.CTkLabel(hdr, text=text.upper(), font=F_CAP, text_color=BLUE,
                     anchor="w").pack(side="left")
        _rule(hdr).pack(side="left", fill="x", expand=True, padx=(8, 0))

    def _build_left(self, p):
        p.grid_rowconfigure(0, weight=1)
        p.grid_columnconfigure(0, weight=1)

        # Pinned primary action (always visible).
        action = ctk.CTkFrame(p, fg_color=PANEL, corner_radius=0)
        action.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 12))
        _rule(action).pack(fill="x", pady=(0, 10))
        self.run_btn = self._btn(action, "RUN PIPELINE", self._run, primary=True, height=42,
                                 font=F_RUN)
        self.run_btn.pack(fill="x")
        self.status_lbl = ctk.CTkLabel(action, text="Ready", font=F_SMALL, text_color=MUTED,
                                       anchor="w", justify="left", wraplength=LEFT_W - 50)
        self.status_lbl.pack(fill="x", pady=(6, 0))

        sc = ctk.CTkScrollableFrame(p, fg_color=PANEL, corner_radius=0,
                                    scrollbar_button_color=BORDER_DARK,
                                    scrollbar_button_hover_color=MUTED)
        sc.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=(0, 4))

        self._section_header(sc, "Input Data")
        self._btn(sc, "Open MRI Slice…", self._load, height=34).pack(fill="x", pady=(0, 4))
        self.file_lbl = ctk.CTkLabel(sc, text="No MRI scan selected", font=F_SMALL,
                                     text_color=FAINT, anchor="w", justify="left", wraplength=LEFT_W - 50)
        self.file_lbl.pack(fill="x")

        self._section_header(sc, "Pipeline Method")
        for val, lab in (("esrg", "ESRG (Enhanced Model)"), ("srg", "SRG Baseline (1994)")):
            self._radio(sc, lab, self.method, val,
                        lambda: (self._update_seed_region_visibility(),
                                 self._controls_changed())).pack(anchor="w", pady=2)

        self._section_header(sc, "Seeding Strategy")
        for val, lab in (("auto", "Automated Seeding"), ("manual", "Manual Seeding")):
            self._radio(sc, lab, self.seed_mode, val,
                        self._on_seed_mode_change).pack(anchor="w", pady=2)

        # Seed region palette + Clear button: clicks are planted as the selected region.
        # Only meaningful for Manual seeding (and the SRG baseline, which tessellates the
        # head between all planted regions — region 2+ is what stops the tumor region from
        # swallowing the whole slice), so it lives in a slot that collapses to nothing in
        # Automated mode (see _update_seed_region_visibility).
        self.region_slot = ctk.CTkFrame(sc, fg_color="transparent", height=1)
        self.region_slot.pack(fill="x")
        self.region_box = ctk.CTkFrame(self.region_slot, fg_color="transparent")
        ctk.CTkLabel(self.region_box, text="Seed region to plant", font=F_SMALL,
                     text_color=MUTED, anchor="w").pack(fill="x", pady=(4, 0))
        for t in range(1, self.cfg.manual_seed_types + 1):
            row = ctk.CTkFrame(self.region_box, fg_color="transparent")
            row.pack(fill="x", pady=1)
            self._radio(row, f"Region {t}" + (" (tumor)" if t == 1 else ""),
                        self.seed_type, t).pack(side="left")
            tk.Frame(row, bg="#%02x%02x%02x" % SEED_RGB[(t - 1) % len(SEED_RGB)],
                     width=14, height=14).pack(side="right", padx=8)
        self._btn(self.region_box, "Clear Manual Seeds", self._clear_seeds, height=28,
                  font=F_SMALL).pack(fill="x", pady=(6, 2))
        self._update_seed_region_visibility()

        self._section_header(sc, "Ablation Controls")
        for var, lab in ((self.use_log, "Log-Domain Transform (Obj 2)"),
                         (self.use_local, "Local Log Measure (Obj 2)"),
                         (self.use_stop, "Adaptive Termination (Obj 3)"),
                         (self.purify_manual_seed, "Purify Manual Seed (ESRG only)")):
            self._check(sc, lab, var, self._controls_changed).pack(anchor="w", pady=3)

        self._section_header(sc, "Hyperparameters")
        self._slider(sc, "k_L — Local Stopping Factor", self.k_local, 0.5, 4.0, 0.1, "{:.1f}",
                     self._controls_changed)
        self._slider(sc, "r — Neighborhood Radius", self.radius, 1, 8, 1, "{:.0f}", self._controls_changed)
        self._slider(sc, "K — Otsu Threshold Classes", self.classes, 2, 5, 1, "{:.0f}",
                     self._controls_changed)

        self._section_header(sc, "Display")
        self._slider(sc, "Overlay Opacity", self.opacity, 0.0, 1.0, 0.05, "{:.2f}",
                     self._schedule_render)
        self._check(sc, "Show Error Heatmap (TP/FP/FN)", self.error_mode,
                    self._schedule_render).pack(anchor="w", pady=(4, 8))

    def _slider(self, p, label, var, lo, hi, step, fmt, cmd=None):
        """Flat track, solid square thumb, live numeric readout."""
        box = ctk.CTkFrame(p, fg_color="transparent")
        box.pack(fill="x", pady=(2, 6))
        head = ctk.CTkFrame(box, fg_color="transparent")
        head.pack(fill="x")
        ctk.CTkLabel(head, text=label, font=F_SMALL, text_color=MUTED,
                     anchor="w").pack(side="left")
        readout = ctk.CTkLabel(head, text=fmt.format(var.get()), font=F_BOLD, text_color=TEXT,
                               anchor="e")
        readout.pack(side="right")

        def changed(v):
            readout.configure(text=fmt.format(v))
            if cmd:
                cmd()

        ctk.CTkSlider(box, from_=lo, to=hi, number_of_steps=int(round((hi - lo) / step)),
                      variable=var, command=changed, height=18, corner_radius=0,
                      button_corner_radius=2, button_length=4, border_width=6,
                      fg_color=BORDER_DARK, progress_color=BLUE, button_color=BLUE,
                      button_hover_color=BLUE_HOV).pack(fill="x", pady=(4, 0))

    def _controls_changed(self):
        """A control moved: refresh the telemetry once (debounced) so its 'controls
        changed since this run' notice appears or clears."""
        if getattr(self, "_ctl_pending", None):
            self.root.after_cancel(self._ctl_pending)
        self._ctl_pending = self.root.after(200, self._refresh_after_controls)

    def _refresh_after_controls(self):
        self._ctl_pending = None
        if self.result is not None:
            self._refresh_telemetry()

    def _on_seed_mode_change(self):
        self._update_seed_region_visibility()
        self._refresh_telemetry()

    def _update_seed_region_visibility(self):
        """The seed-region palette only matters for Manual seeding; automatic SRG
        plants its own background seeds."""
        if self.seed_mode.get() == "manual":
            self.region_box.pack(fill="x")
        else:
            self.region_box.pack_forget()

    # ── Centre column: tabs, twin panes, view controls ───────────────────────
    def _build_center(self, p):
        p.grid_columnconfigure(0, weight=1)
        p.grid_rowconfigure(1, weight=1)

        # Stage tabs. The strip is a horizontally scrolling Canvas rather than a plain
        # Frame: a Frame's natural width (every stage tab + Compare + Evaluation) would
        # otherwise inflate the centre column's requested width and squeeze the other
        # columns. A Canvas's requested size is independent of its scrollable content.
        strip = ctk.CTkFrame(p, fg_color=WHITE, corner_radius=0)
        strip.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 8))
        ctk.CTkLabel(strip, text="STAGE", font=F_CAP, text_color=FAINT).pack(side="left", padx=(0, 10))

        # Scroll arrows keep tabs beyond the visible width discoverable.
        self._btn(strip, "‹", lambda: self._tab_canvas.xview_scroll(-3, "units"), width=26,
                  font=F_BOLD).pack(side="left", padx=(0, 4))
        # Compare view only: superimpose the ESRG mask on the ground truth instead of side by side.
        self.overlay_chk = self._check(strip, "Overlay", self.compare_overlay,
                                       self._on_overlay_toggle)
        self.overlay_chk.pack(side="right", padx=(8, 0))
        self._btn(strip, "›", lambda: self._tab_canvas.xview_scroll(3, "units"), width=26,
                  font=F_BOLD).pack(side="right", padx=(4, 0))
        self._tab_canvas = tk.Canvas(strip, bg=WHITE, height=self._px(32), highlightthickness=0)
        self._tab_canvas.pack(side="left", fill="x", expand=True)
        self.stage_bar = tk.Frame(self._tab_canvas, bg=WHITE)
        self.stage_buttons = {}
        self._tab_canvas.create_window((0, 0), window=self.stage_bar, anchor="nw")
        self.stage_bar.bind("<Configure>", self._on_stage_bar_resize)
        # Resizing the strip (window drag, telemetry collapse, wider Details panel) can push
        # the selected tab out of view; bring it back without disturbing manual scrolling.
        self._tab_canvas.bind("<Configure>", lambda e: self.root.after_idle(self._reveal_current_tab))

        def tab_scroll(event):
            if getattr(event, "num", None) == 4:
                self._tab_canvas.xview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                self._tab_canvas.xview_scroll(1, "units")
            else:
                self._tab_canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")

        self._tab_scroll = tab_scroll
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._tab_canvas.bind(ev, tab_scroll)
        self._init_empty_stage_tabs()

        # Two equal panes: Original | Segmented Output. Each is a caption over a black canvas.
        panes = ctk.CTkFrame(p, fg_color=WHITE, corner_radius=0)
        panes.grid(row=1, column=0, sticky="nsew", padx=14)
        panes.grid_rowconfigure(0, weight=1)
        panes.grid_columnconfigure((0, 1), weight=1, uniform="pane")

        self.captions, self.legends, self.canvases = [], [], []
        for i, title in enumerate(("ORIGINAL", "SEGMENTED OUTPUT")):
            col = ctk.CTkFrame(panes, fg_color=WHITE, corner_radius=0)
            col.grid(row=0, column=i, sticky="nsew", padx=(0, 6) if i == 0 else (6, 0))
            col.grid_rowconfigure(2, weight=1)
            col.grid_columnconfigure(0, weight=1)
            cap = ctk.CTkLabel(col, text=title, font=F_CAP, text_color=TEXT, anchor="w", height=18)
            cap.grid(row=0, column=0, sticky="ew")
            leg = ctk.CTkLabel(col, text="", font=("Segoe UI", 11), text_color=MUTED, anchor="w",
                               height=16)
            leg.grid(row=1, column=0, sticky="ew", pady=(0, 4))
            cv = tk.Canvas(col, bg=VIEWPORT, highlightthickness=0, width=2, height=2)
            cv.grid(row=2, column=0, sticky="nsew")
            self._bind_canvas(cv)
            self.captions.append(cap)
            self.legends.append(leg)
            self.canvases.append(cv)

        # View controls: zoom (- 100% +), Pan tool, Reset.
        ctl = ctk.CTkFrame(p, fg_color=WHITE, corner_radius=0)
        ctl.grid(row=2, column=0, pady=(10, 12))
        self._btn(ctl, "−", self._zoom_out, width=36, font=F_BOLD).pack(side="left")
        self.zoom_lbl = ctk.CTkLabel(ctl, text="100%", font=F_BOLD, text_color=TEXT, width=64)
        self.zoom_lbl.pack(side="left")
        self._btn(ctl, "+", self._zoom_in, width=36, font=F_BOLD).pack(side="left")
        self.pan_btn = self._btn(ctl, "Pan", self._toggle_pan, width=72)
        self.pan_btn.pack(side="left", padx=(18, 0))
        self._btn(ctl, "Reset", self._zoom_reset, width=72).pack(side="left", padx=(6, 0))

    def _bind_canvas(self, c):
        c.bind("<Configure>", lambda e: self._schedule_render())
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>", "<Control-MouseWheel>",
                   "<Control-Button-4>", "<Control-Button-5>"):
            c.bind(ev, self._on_wheel_zoom)
        # A plain left click places a manual seed; a left-drag (or any drag while the
        # Pan tool is on) pans. Right-drag always pans.
        c.bind("<ButtonPress-1>", self._on_left_press)
        c.bind("<B1-Motion>", self._on_left_drag)
        c.bind("<ButtonRelease-1>", self._on_left_release)
        c.bind("<ButtonPress-3>", self._on_pan_start)
        c.bind("<B3-Motion>", self._on_pan_move)

    def _on_stage_bar_resize(self, e):
        self._tab_canvas.configure(scrollregion=self._tab_canvas.bbox("all"),
                                   height=max(e.height, self._px(32)))

    def _init_empty_stage_tabs(self):
        self._clear_stages()
        ctk.CTkLabel(self.stage_bar, text="No stages processed yet", font=F_SMALL,
                     text_color=FAINT, height=30).pack(side="left", padx=4)

    def _toggle_pan(self):
        self.pan_tool.set(not self.pan_tool.get())
        self._style_toggle(self.pan_btn, self.pan_tool.get())
        for c in self.canvases:
            c.configure(cursor="fleur" if self.pan_tool.get() else "")

    def _sync_overlay_toggle(self):
        """Overlay needs a ground truth to superimpose; otherwise it is greyed out and cleared."""
        usable = self.result is not None and self.result.gt is not None
        if not usable:
            self.compare_overlay.set(False)
        self.overlay_chk.configure(state="normal" if usable else "disabled")

    def _on_overlay_toggle(self):
        # Overlay only changes the Compare view, so ticking it from another tab opens Compare.
        if self.compare_overlay.get() and self.current != "compare":
            self._select("compare")
        else:
            self._render_panes()

    # ── Right column: collapsible seed telemetry ─────────────────────────────
    def _build_telemetry(self, p):
        p.grid_columnconfigure(0, weight=1)
        p.grid_rowconfigure(2, weight=1)

        head = ctk.CTkFrame(p, fg_color=WHITE, corner_radius=0)
        head.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        ctk.CTkLabel(head, text="STAGE TELEMETRY", font=F_CAP, text_color=BLUE).pack(side="left")
        self._btn(head, ">|", self._toggle_telemetry, width=36, height=26,
                  font=F_CAP).pack(side="right")

        modes = ctk.CTkFrame(p, fg_color=WHITE, corner_radius=0)
        modes.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        self.tel_btns = {}
        for name in ("Seeds", "Details"):
            b = self._btn(modes, name, lambda n=name: self._set_tel_mode(n), width=84, height=26,
                          font=F_SMALL)
            b.pack(side="left", padx=(0, 4))
            self.tel_btns[name] = b
        self._style_toggle(self.tel_btns["Seeds"], True)

        body = ctk.CTkFrame(p, fg_color=WHITE, corner_radius=0)
        body.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)

        # Seeds view: flat table (+ empty-state message shown in its place).
        self.seed_view = ctk.CTkFrame(body, fg_color=WHITE, corner_radius=0)
        self.seed_view.grid_rowconfigure(0, weight=1)
        self.seed_view.grid_columnconfigure(0, weight=1)
        cols = ("row", "col", "raw", "norm", "status")
        self.tree = ttk.Treeview(self.seed_view, columns=cols, style="Seeds.Treeview",
                                 selectmode="browse")
        self.tree.heading("#0", text="Seed", anchor="w")
        self.tree.column("#0", width=self._px(112), minwidth=self._px(80), stretch=True)
        for key, text, w, anchor in (("row", "Row", 40, "e"), ("col", "Col", 40, "e"),
                                     ("raw", "Raw", 46, "e"), ("norm", "Norm", 56, "e"),
                                     ("status", "Status", 88, "w")):
            self.tree.heading(key, text=text, anchor=anchor)
            self.tree.column(key, width=self._px(w), minwidth=self._px(w), anchor=anchor,
                             stretch=key == "status")
        self.tree.tag_configure("group", background=PANEL, font=TK_BOLD)
        self.tree.tag_configure("good", foreground=OK_CLR)
        self.tree.tag_configure("warn", foreground=WARN_CLR)
        self.tree.tag_configure("bad", foreground=FAIL_CLR)
        self.tree.tag_configure("muted", foreground=MUTED)
        sb = ctk.CTkScrollbar(self.seed_view, command=self.tree.yview, corner_radius=0,
                              fg_color=WHITE, button_color=BORDER_DARK,
                              button_hover_color=MUTED, width=12)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        ctk.CTkLabel(self.seed_view, anchor="w", justify="left", font=("Segoe UI", 11),
                     text_color=MUTED, wraplength=340,
                     text="row = y, col = x (0-based) · Raw = file value · Norm = value the "
                          "algorithm uses (0–255) · group row = mean of its pixels"
                     ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.seed_msg = ctk.CTkLabel(body, text="", font=F_SMALL, text_color=MUTED, anchor="nw",
                                     justify="left", wraplength=330)

        # Details view: metric picker (Evaluation only) over the worked-computation text.
        self.detail_view = ctk.CTkFrame(body, fg_color=WHITE, corner_radius=0)
        self.detail_view.grid_rowconfigure(1, weight=1)
        self.detail_view.grid_columnconfigure(0, weight=1)
        self.eval_picker = ctk.CTkFrame(self.detail_view, fg_color=WHITE, corner_radius=0)
        pick_head = ctk.CTkFrame(self.eval_picker, fg_color=WHITE, corner_radius=0)
        pick_head.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(pick_head, text="Select metrics to compute:", font=F_SMALL,
                     text_color=MUTED).pack(side="left")
        for text, val in (("None", False), ("All", True)):
            self._btn(pick_head, text, lambda v=val: self._set_all_metrics(v), width=46, height=22,
                      font=F_SMALL).pack(side="right", padx=(4, 0))
        grid = ctk.CTkFrame(self.eval_picker, fg_color=WHITE, corner_radius=0)
        grid.pack(fill="x", pady=(0, 8))
        for i, (key, label, _) in enumerate(evalx.METRICS):
            self._check(grid, label, self.eval_vars[key], self._refresh_telemetry).grid(
                row=i // 3, column=i % 3, sticky="w", padx=(0, 10), pady=2)

        text_box = ctk.CTkFrame(self.detail_view, fg_color=WHITE, corner_radius=0)
        text_box.grid(row=1, column=0, sticky="nsew")
        text_box.grid_rowconfigure(0, weight=1)
        text_box.grid_columnconfigure(0, weight=1)
        self.diag = tk.Text(text_box, bg=PANEL, fg=TEXT, font=TK_MONO, wrap=tk.WORD,
                            relief=tk.FLAT, bd=0, highlightthickness=1,
                            highlightbackground=BORDER, highlightcolor=BORDER, padx=10, pady=10)
        dsb = ctk.CTkScrollbar(text_box, command=self.diag.yview, corner_radius=0, fg_color=WHITE,
                               button_color=BORDER_DARK, button_hover_color=MUTED, width=12)
        self.diag.configure(yscrollcommand=dsb.set)
        self.diag.grid(row=0, column=0, sticky="nsew")
        dsb.grid(row=0, column=1, sticky="ns")
        self.diag.tag_config("head", foreground=BLUE, font=TK_MONO + ("bold",))
        self.diag.tag_config("muted", foreground=MUTED)
        self.diag.tag_config("avg", foreground=TEXT, font=TK_MONO + ("bold",))
        self.diag.tag_config("metric", foreground=TEXT, font=(TK_MONO[0], 11, "bold"))
        self.diag.tag_config("sub", foreground=BLUE, font=TK_MONO + ("bold",))
        self.diag.tag_config("good", foreground=OK_CLR, font=TK_MONO + ("bold",))
        self.diag.tag_config("bad", foreground=FAIL_CLR, font=TK_MONO + ("bold",))
        self.diag.tag_config("indent", lmargin1=18, lmargin2=18)
        # Stage report blocks (esrg/stage_report.py).
        self.diag.tag_config("flow", foreground=FAINT, font=TK_UI)
        self.diag.tag_config("flow_cur", foreground=WHITE, background=BLUE, font=TK_BOLD)
        self.diag.tag_config("stage", foreground=TEXT, font=(TK_UI[0], 12, "bold"),
                             spacing1=6, spacing3=2)
        self.diag.tag_config("section", foreground=BLUE, font=TK_BOLD, spacing1=10, spacing3=3)
        self.diag.tag_config("in_band", background=IN_BG, lmargin1=8, lmargin2=8, rmargin=4)
        self.diag.tag_config("out_band", background=OUT_BG, lmargin1=8, lmargin2=8, rmargin=4)
        self.diag.tag_config("in_lbl", foreground=BLUE, font=TK_MONO + ("bold",))
        self.diag.tag_config("out_lbl", foreground=OK_CLR, font=TK_MONO + ("bold",))
        self.diag.tag_config("eq", background=WHITE, foreground=TEXT, font=TK_MONO + ("bold",),
                             lmargin1=12, lmargin2=12, spacing1=1, spacing3=1)
        self.diag.tag_config("table", wrap=tk.NONE, lmargin1=4)
        self.diag.tag_config("thead", foreground=MUTED, font=TK_MONO + ("bold",), underline=True)
        self.diag.tag_config("note", foreground=MUTED, font=TK_UI, lmargin1=4, lmargin2=4)
        self.diag.tag_config("changed", foreground=WARN_CLR, font=TK_MONO + ("bold",))
        self.diag.tag_config("stale", foreground=WHITE, background=WARN_CLR, font=TK_BOLD,
                             lmargin1=6, lmargin2=6, rmargin=6, spacing1=4, spacing3=4)

        self._refresh_telemetry()

    def _toggle_telemetry(self):
        """Hide / show the telemetry column. The centre column holds the only grid weight,
        so it reclaims (or yields) the freed width on its own."""
        self.tel_visible = not self.tel_visible
        if self.tel_visible:
            _hide(self.rail)
            self.tel_panel.grid(row=0, column=4, sticky="ns")
        else:
            _hide(self.tel_panel)
            self.rail.grid(row=0, column=4, sticky="ns")
        # Recompute geometry now so the redraw already sees the new canvas size.
        self.root.update_idletasks()
        self._render_panes()

    def _set_tel_mode(self, mode):
        self.tel_mode.set(mode)
        for name, b in self.tel_btns.items():
            self._style_toggle(b, name == mode)
        self._refresh_telemetry()

    def _telemetry_groups(self):
        """(groups, None) when there are seed pixels to list, else (None, message)."""
        if self.raw_image is None:
            return None, "Load an MRI slice to inspect its seed pixels."
        if self.seed_mode.get() == "manual":
            groups = pixel_report.manual_groups(self.manual_points)
            if not groups:
                return None, "Click on the image to select seed pixels."
            return groups, None
        # The last run may have been made in Manual mode; its core is not a
        # system selection, so only an auto run is shown here.
        if not self.result or self.result.meta.get("seed_mode") != "auto":
            return None, "Run the pipeline in Automated mode to see the pixels the system selects."
        groups = pixel_report.auto_groups(self.result.stage("seed").image)
        if not groups:
            return None, "No tumor candidate — the system selected no seed pixels."
        return groups, None

    def _seed_status(self, label, r, c):
        """(text, tag) status of one seed pixel, derived only from what the last run did."""
        res = self.result
        if self.seed_mode.get() == "auto":
            if res.gt is not None:
                return ("In tumor", "good") if res.gt[r, c] else ("Off tumor", "bad")
            return "Active", "good"
        region = int(label.split()[1])
        ran = res is not None and res.meta.get("seed_mode") == "manual"
        if not ran or (r, c, region) not in self._run_points:
            return "Placed", "muted"       # no run yet, or clicked after the last run
        if res.meta.get("method") != "esrg":
            return "Active", "good"        # SRG grows every planted region over the whole image
        # ESRG grows only Region 1, inside the head mask.
        if region != 1:
            return "Ignored", "muted"
        if not res.stage("mask").image[r, c]:
            return "Outside head", "bad"
        if res.stage("seed").image[r, c]:
            return "Active", "good"
        return "Purified out", "warn"

    def _refresh_telemetry(self):
        details = self.tel_mode.get() == "Details"
        evaluating = details and self.result is not None and self.current == "evaluation"
        self._sync_eval_picker(evaluating)
        if details:
            _hide(self.seed_view)
            _hide(self.seed_msg)
            self.detail_view.grid(row=0, column=0, sticky="nsew")
            self._fill_details()
        else:
            _hide(self.detail_view)
            self._fill_seed_table()

    def _fill_seed_table(self):
        self.tree.delete(*self.tree.get_children())
        groups, msg = self._telemetry_groups()
        if msg:
            _hide(self.seed_view)
            self.seed_msg.configure(text=msg)
            self.seed_msg.grid(row=0, column=0, sticky="nsew", padx=2, pady=4)
            return
        _hide(self.seed_msg)
        self.seed_view.grid(row=0, column=0, sticky="nsew")
        for g in pixel_report.describe(groups, self.raw_image, self.norm_image):
            parent = self.tree.insert(
                "", "end", text=g["label"], open=True, tags=("group",),
                values=("", "", f"{g['mean_raw']:.1f}", f"{g['mean_norm']:.2f}", f"{g['n']} px"))
            for i, p in enumerate(g["pixels"], 1):
                status, tag = self._seed_status(g["label"], p["row"], p["col"])
                self.tree.insert(parent, "end", text=str(i), tags=(tag,),
                                 values=(p["row"], p["col"], f"{p['raw']:.0f}",
                                         f"{p['norm']:.2f}", status))

    def _fill_details(self):
        d = self.diag
        top = d.yview()[0]
        d.config(state=tk.NORMAL)
        d.delete("1.0", tk.END)
        if self.result is None:
            d.insert(tk.END, "Run the pipeline to see the worked computation of each stage.\n",
                     "muted")
        else:
            run_cfg = self.result.meta.get("_cfg") or self.run_cfg
            if run_cfg is not None and run_cfg != self._current_config():
                d.insert(tk.END, "Controls changed since this run. The values below are for the run as "
                                 "executed; press RUN PIPELINE to apply the new settings.\n", "stale")
            chosen = [k for k, _, _ in evalx.METRICS if self.eval_vars[k].get()]
            self._render_blocks(stage_report.report(self.result, self.run_cfg or self.cfg,
                                                    self.current, chosen))
        d.yview_moveto(top)
        d.config(state=tk.DISABLED)

    def _render_blocks(self, blocks):
        """Draw stage_report blocks: flow strip, INPUT / OUTPUT bands, equations, tables."""
        d = self.diag
        for kind, p in blocks:
            if kind == "flow":
                for i, (_, name, cur) in enumerate(p):
                    if i:
                        d.insert(tk.END, " → ", "flow")
                    d.insert(tk.END, f" {name} " if cur else name, "flow_cur" if cur else "flow")
                d.insert(tk.END, "\n")
            elif kind == "head":
                d.insert(tk.END, p + "\n", "stage")
            elif kind == "sub":
                d.insert(tk.END, p + "\n", "section")
            elif kind in ("input", "output"):
                band, lbl = ("in_band", "in_lbl") if kind == "input" else ("out_band", "out_lbl")
                word, prep = ("IN ", "from") if kind == "input" else ("OUT", "to")
                for name, value, where in p:
                    d.insert(tk.END, f"{word} ", (band, lbl))
                    d.insert(tk.END, f"{name} = ", (band, "avg"))
                    d.insert(tk.END, f"{value}\n", band)
                    d.insert(tk.END, f"    {prep}: {where}\n", (band, "muted"))
            elif kind == "eq":
                for line in p:
                    d.insert(tk.END, f" {line} \n", "eq")
            elif kind == "kv":
                for k, v in p:
                    d.insert(tk.END, f"{k}: ", "muted")
                    d.insert(tk.END, f"{v}\n")
            elif kind == "settings":
                d.insert(tk.END, "SETTINGS USED IN THIS RUN\n", "section")
                for label, value, changed, note in p:
                    d.insert(tk.END, f"{label}: ", "muted")
                    d.insert(tk.END, value, "changed" if changed else "avg")
                    d.insert(tk.END, f"   {note}\n" if note else "\n", "muted")
            elif kind == "table":
                self._render_table(p)
            elif kind == "note":
                d.insert(tk.END, p + "\n", "note")
            elif kind in ("good", "bad"):
                d.insert(tk.END, p + "\n", kind)
            else:
                d.insert(tk.END, str(p) + "\n")

    @staticmethod
    def _is_number(v):
        t = v.replace(",", "").replace("−", "-").replace(" px", "").lstrip("+-")
        try:
            float(t)
            return True
        except ValueError:
            return False

    def _render_table(self, t):
        """Monospace table: numeric columns right-aligned, text left-aligned, header underlined."""
        d = self.diag
        cols = list(t["cols"])
        rows = [[str(c) for c in r] for r in t["rows"]]
        n = len(cols)
        w = [max([len(cols[i])] + [len(r[i]) for r in rows]) for i in range(n)]
        num = [bool(rows) and all(self._is_number(r[i]) or r[i] in ("", "—") for r in rows)
               for i in range(n)]

        def line(cells):
            return "  ".join(c.rjust(w[i]) if num[i] else c.ljust(w[i]) for i, c in enumerate(cells))

        d.insert(tk.END, line(cols) + "\n", ("table", "thead"))
        for r in rows:
            d.insert(tk.END, line(r) + "\n", "table")
        if t.get("note"):
            d.insert(tk.END, t["note"] + "\n", "note")
        d.insert(tk.END, "\n")

    def _sync_eval_picker(self, evaluating):
        """Widen the panel in Details mode; show the metric picker only for the Evaluation step."""
        details = self.tel_mode.get() == "Details"
        self.tel_panel.configure(width=DETAIL_PANEL_W if details else DIAG_PANEL_W)
        if evaluating and not self.eval_picker.winfo_ismapped():
            self.eval_picker.grid(row=0, column=0, sticky="ew")
        elif not evaluating:
            _hide(self.eval_picker)

    def _set_all_metrics(self, value):
        for v in self.eval_vars.values():
            v.set(value)
        self._refresh_telemetry()

    # ── Footer: one-line metrics ─────────────────────────────────────────────
    def _build_footer(self, parent):
        bar = ctk.CTkFrame(parent, fg_color=WHITE, corner_radius=0, height=42)
        bar.grid(row=4, column=0, sticky="ew")
        bar.pack_propagate(False)
        self.metric_vals = {}
        for i, (label, key, _) in enumerate(FOOTER_METRICS):
            if i:
                _rule(bar, True).pack(side="left", fill="y", pady=10)
            cell = ctk.CTkFrame(bar, fg_color=WHITE, corner_radius=0)
            cell.pack(side="left", padx=18)
            ctk.CTkLabel(cell, text=label, font=F_SMALL, text_color=MUTED).pack(side="left")
            val = ctk.CTkLabel(cell, text="—", font=F_BOLD, text_color=FAINT)
            val.pack(side="left", padx=(8, 0))
            self.metric_vals[key] = val
        self.foot_msg = ctk.CTkLabel(bar, text="", font=F_SMALL, text_color=MUTED, anchor="e")
        self.foot_msg.pack(side="right", padx=18)

    def _update_metrics(self):
        res = self.result
        scores = res.scores if res else {}
        for _, key, fmt in FOOTER_METRICS:
            v = scores.get(key)
            ok = v is not None and not (isinstance(v, float) and np.isnan(v))
            self.metric_vals[key].configure(text=fmt.format(v) if ok else "—",
                                            text_color=TEXT if ok else FAINT)
        if res is None:
            msg, color = (("Load an MRI slice to begin.", MUTED) if self.raw_image is None
                          else ("Slice loaded — press RUN PIPELINE.", MUTED))
        elif res.status == "NO TUMOR CANDIDATE":
            msg, color = "NO TUMOR CANDIDATE — seeds were filtered during interior checks.", FAIL_CLR
        elif not scores:
            msg, color = "No ground truth found — metrics unavailable.", WARN_CLR
        else:
            msg, color = f"Status: {res.status} · scored against the ground-truth mask", MUTED
        self.foot_msg.configure(text=msg, text_color=color)

    # ── Controller & Backend Invocation ──────────────────────────────────────
    def _load(self):
        path = filedialog.askopenfilename(
            title="Select MRI Slice",
            filetypes=[("Medical Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All Files", "*.*")])
        if not path:
            return
        self.load_path(path)

    def load_path(self, path):
        self.image_path = path
        self.result = None
        self.manual_points = []
        self._run_points = set()

        try:
            self.raw_image, _ = load_image(path, self.cfg.max_side)
            self.norm_image = pre.normalize(self.raw_image)
        except Exception as e:
            self.raw_image = None
            self.norm_image = None
            self._refresh_telemetry()
            messagebox.showerror("Load Failed", f"Could not read this image:\n{e}")
            return

        self._zoom_reset()
        self._set_tel_mode("Seeds")

        gt = find_mask_for(path)
        self.file_lbl.configure(
            text=f"{os.path.basename(path)}\n"
                 f"• {'Ground truth detected' if gt else 'No ground truth found'}",
            text_color=MUTED)
        self._init_empty_stage_tabs()
        self._sync_overlay_toggle()
        self._sync_obj_buttons()
        self._update_metrics()
        self._render_panes()

    def _clear_seeds(self):
        self.manual_points = []
        self.status_lbl.configure(text="Manual seed coordinates cleared.", text_color=MUTED)
        self._refresh_telemetry()
        self._render_panes()

    def _on_click(self, event):
        if self.seed_mode.get() != "manual" or not self.image_path:
            return
        rc = self._canvas_to_image(event.x, event.y, event.widget)
        if rc:
            self.manual_points.append((rc[0], rc[1], self.seed_type.get()))
            n_regions = len({p[2] for p in self.manual_points})
            self.status_lbl.configure(
                text=f"{len(self.manual_points)} seed(s) across {n_regions} region(s) placed.",
                text_color=BLUE)
            self._refresh_telemetry()
            self._render_panes()

    def _current_config(self):
        return self.cfg.replace(
            method=self.method.get(), seed_mode=self.seed_mode.get(),
            use_log_local=self.use_local.get(), use_stopping=self.use_stop.get(),
            use_log=self.use_log.get(), purify_manual_seed=self.purify_manual_seed.get(),
            k_local=round(float(self.k_local.get()), 1),
            k_global=max(round(float(self.k_local.get()), 1) + 1.0, 3.0),
            local_radius=int(self.radius.get()), otsu_classes=int(self.classes.get()))

    def _run(self):
        if not self.image_path:
            messagebox.showwarning("Input Required", "Please load an MRI slice before running.")
            return
        if self.seed_mode.get() == "manual" and not any(p[2] == 1 for p in self.manual_points):
            messagebox.showwarning("Seed Required",
                                   "Click on the tumor region to place at least one Region 1 seed.")
            return

        cfg = self._current_config()
        self.run_cfg = cfg
        self._run_points = {tuple(p) for p in self.manual_points}
        self.run_btn.configure(state="disabled", text="PROCESSING…", fg_color=BORDER_DARK,
                               border_color=BORDER_DARK)
        self.status_lbl.configure(text="Segmenting slice…", text_color=BLUE)

        def worker():
            try:
                res = run(self.image_path, cfg, manual_points=list(self.manual_points),
                          record=True, progress=lambda m: self.root.after(
                              0, lambda m=m: self.status_lbl.configure(text=m)))
                self.root.after(0, self._done, res, None)
            except Exception as e:
                self.root.after(0, self._done, None, e)

        threading.Thread(target=worker, daemon=True).start()

    def _done(self, res, err):
        self.run_btn.configure(state="normal", text="RUN PIPELINE", fg_color=BLUE,
                               border_color=BLUE)
        if err:
            self.status_lbl.configure(text="Pipeline execution failed.", text_color=FAIL_CLR)
            messagebox.showerror("Execution Error", str(err))
            return

        self.result = res
        self.status_lbl.configure(text=f"Status: {res.status}",
                                  text_color=OK_CLR if res.status == "OK" else WARN_CLR)
        self._build_stage_buttons()
        self._sync_overlay_toggle()
        self._update_metrics()
        self._select("final")

    # ── Stage tabs ───────────────────────────────────────────────────────────
    def _clear_stages(self):
        for w in self.stage_bar.winfo_children():
            w.destroy()
        self.stage_buttons = {}

    def _add_tab(self, key, text):
        btn = ctk.CTkButton(self.stage_bar, text=text, command=lambda k=key: self._select(k),
                            height=30, corner_radius=0, border_width=0, font=F_SMALL,
                            fg_color=PANEL, hover_color=BORDER, text_color=TEXT)
        btn.pack(side="left", padx=(0, 2))
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            btn.bind(ev, self._tab_scroll, add="+")
        self.stage_buttons[key] = btn

    def _build_stage_buttons(self):
        self._clear_stages()
        for st in self.result.stages:
            self._add_tab(st.key, st.name)
        if self.result.gt is not None:
            self._add_tab("compare", "Compare")
        # Always offered after a run; without a ground truth the Details view
        # explains what is missing.
        self._add_tab("evaluation", "Evaluation")

    def _select(self, key):
        self.current = key
        status_colors = {"WARN": WARN_CLR, "FAIL": FAIL_CLR}
        for k, b in self.stage_buttons.items():
            if k == key:
                b.configure(fg_color=BLUE, hover_color=BLUE, text_color=WHITE)
            else:
                st = self.result.stage(k)
                b.configure(fg_color=PANEL, hover_color=BORDER,
                            text_color=status_colors.get(st.status, TEXT) if st else TEXT)

        self._sync_obj_buttons()
        # Every stage tab is explained in Details (Seeds stays one click away). Switching
        # views can resize the telemetry panel, so settle that before scrolling the tab in.
        self._set_tel_mode("Details")
        self._scroll_tab_into_view(self.stage_buttons.get(key))
        self._render_panes()

    def _reveal_current_tab(self):
        if self.result is not None:
            self._scroll_tab_into_view(self.stage_buttons.get(self.current))

    def _scroll_tab_into_view(self, btn):
        """Scroll the strip so the selected tab (e.g. Evaluation, the last one) is
        visible when the strip is narrower than its tabs."""
        if btn is None:
            return
        self.root.update_idletasks()
        total = max(self.stage_bar.winfo_reqwidth(), 1)
        view_w = self._tab_canvas.winfo_width()
        left, right = btn.winfo_x(), btn.winfo_x() + btn.winfo_width()
        x0 = self._tab_canvas.xview()[0] * total
        if left < x0:
            self._tab_canvas.xview_moveto(left / total)
        elif right > x0 + view_w:
            self._tab_canvas.xview_moveto(max(0, right - view_w) / total)

    # ── Zoom & Pan ────────────────────────────────────────────────────────────
    def _has_image(self):
        return self.result is not None or self.raw_image is not None

    def _set_zoom(self, z):
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, z))
        self.zoom_lbl.configure(text=f"{round(self.zoom * 100)}%")
        self._render_panes()

    def _zoom_in(self):
        if self._has_image():
            self._set_zoom(self.zoom * 1.25)

    def _zoom_out(self):
        if self._has_image():
            self._set_zoom(self.zoom / 1.25)

    def _zoom_reset(self):
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._set_zoom(1.0)

    def _on_wheel_zoom(self, event):
        if not self._has_image():
            return
        canvas = event.widget
        zoom_in = not (getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0)
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom * 1.25 if zoom_in else self.zoom / 1.25))

        # Anchor the zoom on the cursor/pinch position: find the image-space
        # point under the cursor before changing zoom, then solve pan so
        # that same point stays under the cursor afterwards. Both panes share
        # identical w/h/s, so solving in whichever one the cursor is over and
        # writing the result into the shared pan keeps them in lockstep.
        g = self._geometry(canvas)
        if g and new_zoom != self.zoom:
            w, h, s_old, ox_old, oy_old = g
            ix = (event.x - ox_old) / s_old
            iy = (event.y - oy_old) / s_old
            cw = max(canvas.winfo_width(), 50)
            ch = max(canvas.winfo_height(), 50)
            s_new = min(cw / w, ch / h) * new_zoom
            self.pan_x = event.x - ix * s_new - (cw - w * s_new) / 2
            self.pan_y = event.y - iy * s_new - (ch - h * s_new) / 2

        self._set_zoom(new_zoom)

    def _on_pan_start(self, event):
        if self._has_image():
            self._pan_start = (event.x, event.y, self.pan_x, self.pan_y)

    def _on_pan_move(self, event):
        if not self._pan_start:
            return
        sx, sy, px, py = self._pan_start
        self.pan_x = px + (event.x - sx)
        self.pan_y = py + (event.y - sy)
        self._render_panes()

    # Left-click-and-hold drag pans (a trackpad has no natural right-click-drag
    # gesture), while a plain click with no real movement places a manual seed.
    # With the Pan tool on, every left press is a drag from the start.
    _DRAG_THRESHOLD = 4

    def _on_left_press(self, event):
        if self._has_image():
            self._left_press = (event.x, event.y, self.pan_x, self.pan_y)
            self._left_dragging = self.pan_tool.get()

    def _on_left_drag(self, event):
        if not self._left_press:
            return
        sx, sy, px, py = self._left_press
        dx, dy = event.x - sx, event.y - sy
        if not self._left_dragging and (abs(dx) + abs(dy)) > self._DRAG_THRESHOLD:
            self._left_dragging = True
        if self._left_dragging:
            self.pan_x = px + dx
            self.pan_y = py + dy
            self._render_panes()

    def _on_left_release(self, event):
        if self._left_press and not self._left_dragging:
            self._on_click(event)
        self._left_press = None
        self._left_dragging = False

    # ── Rendering & Visual Geometry ──────────────────────────────────────────
    def _geometry(self, canvas=None):
        canvas = canvas or self.canvases[0]
        if self.result:
            h, w = self.result.stage("input").image.shape
        elif self.raw_image is not None:
            h, w = self.raw_image.shape
        else:
            return None
        cw = max(canvas.winfo_width(), 50)
        ch = max(canvas.winfo_height(), 50)
        s = min(cw / w, ch / h) * self.zoom
        ox = (cw - w * s) / 2 + self.pan_x
        oy = (ch - h * s) / 2 + self.pan_y
        return w, h, s, ox, oy

    def _canvas_to_image(self, x, y, canvas=None):
        g = self._geometry(canvas)
        if not g:
            return None
        w, h, s, ox, oy = g
        c, r = int((x - ox) / s), int((y - oy) / s)
        return (r, c) if 0 <= r < h and 0 <= c < w else None

    def _schedule_render(self):
        """Coalesce bursts of <Configure> / slider events into one repaint."""
        if not self._render_pending:
            self._render_pending = True
            self.root.after_idle(self._render_panes)

    @staticmethod
    def _mask_rgb(base_img, mask, color, opacity):
        rgb = _gray_rgb(base_img)
        m = mask.astype(bool)
        if m.any():
            rgb[m] = (np.array(color) * opacity + rgb[m] * (1 - opacity)).astype(np.uint8)
        return rgb

    def _evaluation_rgb(self):
        """The pixels each metric counts: TP green, FP red, FN orange (or the
        prediction alone without a ground truth), with the seed core S of
        Objective 1 in cyan."""
        res = self.result
        base = res.stage("input").image
        gt = res.gt
        rgb = viz.overlay_result(base, res.mask.astype(bool), gt, self.opacity.get(),
                                 error_mode=gt is not None)
        seed = res.stage("seed")
        if seed is not None and seed.image.any():
            core = seed.image.astype(bool)
            # Small cores are filled so they stay visible; larger ones are outlined.
            rgb[core if core.sum() < 30 else viz.outline(core)] = (34, 211, 238)
        return rgb

    def _stamp_seeds(self, rgb):
        if self.seed_mode.get() == "manual":
            for r, c, t in self.manual_points:
                rgb[max(0, r - 2):r + 3, max(0, c - 2):c + 3] = SEED_RGB[(t - 1) % len(SEED_RGB)]
        return rgb

    def _pane_images(self):
        """(left_rgb, right_rgb, left_caption, right_caption, right_legend).
        An rgb of None means 'draw the empty-state hint'. Cached across pan / zoom /
        resize, since only the placement changes there."""
        key = (id(self.result), self.current, round(self.opacity.get(), 3),
               self.error_mode.get(), self.seed_mode.get(), id(self.manual_points),
               len(self.manual_points), id(self.raw_image), self.compare_overlay.get())
        if self._pane_cache[0] == key:
            return self._pane_cache[1]

        res, op = self.result, self.opacity.get()
        legend = ""
        if self.raw_image is None and res is None:
            out = (None, None, "ORIGINAL", "SEGMENTED OUTPUT", "")
        elif res is None:
            # Raw scan preview, so the user can confirm the right slice was loaded.
            out = (self._stamp_seeds(_gray_rgb(self.raw_image)), None, "ORIGINAL",
                   "SEGMENTED OUTPUT", "")
        elif self.current == "compare" and res.gt is not None:
            base = res.stage("input").image
            if self.compare_overlay.get():
                # Both masks superimposed on one pane; the other keeps the plain slice for reference.
                legend = ("TP green · FP red · FN orange" if self.error_mode.get()
                          else "ESRG prediction red fill · ground truth green outline")
                out = (_gray_rgb(self.raw_image if self.raw_image is not None else base),
                       viz.overlay_result(base, res.mask, res.gt, op, self.error_mode.get()),
                       "ORIGINAL", "OVERLAY · ESRG vs GROUND TRUTH", legend)
            else:
                out = (self._mask_rgb(base, res.mask, viz.RED, op),
                       self._mask_rgb(base, res.gt, viz.GREEN, op),
                       "ESRG PREDICTION", "GROUND TRUTH", "")
        else:
            left = _gray_rgb(self.raw_image if self.raw_image is not None
                             else res.stage("input").image)
            if self.current == "evaluation":
                right, title = self._evaluation_rgb(), "Evaluation"
                legend = ("TP green · FP red · FN orange · seed core cyan" if res.gt is not None
                          else "Prediction red · seed core cyan (no ground truth)")
            else:
                st = res.stage(self.current) or res.stages[-1]
                base = res.stage("input").image
                if st.key == "final":
                    right = viz.overlay_result(base, st.image, res.gt, op, self.error_mode.get())
                    if res.gt is not None:
                        legend = ("TP green · FP red · FN orange" if self.error_mode.get()
                                  else "Prediction red fill · ground truth green outline")
                else:
                    right = viz.render_stage(st, base, res.gt, op)
                title = st.name
            if self.current != "evaluation":
                self._stamp_seeds(left)
                self._stamp_seeds(right)
            out = (left, right, "ORIGINAL", f"SEGMENTED OUTPUT · {title}", legend)
        self._pane_cache = (key, out)
        return out

    def _render_panes(self):
        self._render_pending = False
        if not hasattr(self, "canvases"):
            return
        left, right, cap_l, cap_r, legend = self._pane_images()
        self.captions[0].configure(text=cap_l)
        self.captions[1].configure(text=cap_r)
        self.legends[1].configure(text=legend)
        hints = ("Open an MRI slice to begin\nScroll to zoom · drag to pan",
                 "The segmented output appears here\nafter RUN PIPELINE")
        for cv, rgb, hint in zip(self.canvases, (left, right), hints):
            self._paint(cv, rgb, hint)

    def _paint(self, canvas, rgb, hint):
        canvas.delete("all")
        g = self._geometry(canvas) if rgb is not None else None
        if g is None:
            canvas.create_text(max(canvas.winfo_width(), 50) // 2, max(canvas.winfo_height(), 50) // 2,
                               text=hint, fill=ON_BLACK, font=TK_UI, justify=tk.CENTER)
            self._photos.pop(canvas, None)
            return
        w, h, s, ox, oy = g
        img = Image.fromarray(rgb.astype(np.uint8)).resize(
            (max(1, int(w * s)), max(1, int(h * s))), Image.NEAREST)
        self._photos[canvas] = ImageTk.PhotoImage(img)  # keep a reference alive
        canvas.create_image(ox, oy, anchor=tk.NW, image=self._photos[canvas])


if __name__ == "__main__":
    root = ctk.CTk()
    ESRGApp(root)
    root.mainloop()
