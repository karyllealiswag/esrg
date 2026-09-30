"""
app.py — Desktop GUI for the ESRG pipeline.

Purpose : Load an MRI slice, run the pipeline, inspect any stage, and read
          evaluation scores with a modern clinical workstation interface.
Function : Tkinter app with a clinical control sidebar, a high-contrast MRI
          viewer with stage navigation tabs, a diagnostics inspector, and a
          results metric bar. Runs off the UI thread and exports all stage masks.
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esrg import Config, run
from esrg import pixel_report
from esrg import preprocessing as pre
from esrg import visualize as viz
from esrg.io_utils import find_mask_for, load_image

# ── Clinical Light Theme Tokens ──────────────────────────────────────────────
APP_BG      = "#f1f5f9"  
PANEL_BG    = "#ffffff"  
PANEL_ALT   = "#f8fafc"  
BORDER_CLR  = "#e2e8f0"  
BORDER_MED  = "#cbd5e1"  

TEXT_MAIN   = "#0f172a"  
TEXT_MUTED  = "#475569"  
TEXT_FAINT  = "#94a3b8"  

PRIMARY     = "#0284c7"  
PRIMARY_HOV = "#0369a1"  
VIEWPORT_BG = "#090d16"  

# Status Badges
OK_CLR      = "#16a34a"
WARN_CLR    = "#d97706"
FAIL_CLR    = "#dc2626"

# Manual seed regions share the tessellation palette (esrg.visualize.SEED_COLORS):
# region 1 is whichever region the user treats as the structure of interest,
# region 2+ are the other competing regions. SRG itself does not distinguish them.
SEED_RGB = viz.SEED_COLORS

ZOOM_MIN = 0.25
ZOOM_MAX = 8.0

# Typography
FONT_TITLE  = ("Segoe UI", 11, "bold")
FONT_SUB    = ("Segoe UI", 9)
FONT_BOLD   = ("Segoe UI", 8, "bold")
FONT_UI     = ("Segoe UI", 9)
FONT_SM     = ("Segoe UI", 8)
FONT_MONO   = ("Consolas", 9) if sys.platform == "win32" else ("Menlo", 8)


# ── Custom Cross-Platform Flat Button ────────────────────────────────────────
class FlatButton(tk.Label):
    def __init__(self, master, text, command, bg, fg, hover_bg, **kwargs):
        self.border_clr = kwargs.pop('border_color', BORDER_CLR)
        kwargs.setdefault('highlightthickness', 1)
        kwargs.setdefault('highlightbackground', self.border_clr)
        
        super().__init__(master, text=text, bg=bg, fg=fg, cursor="hand2", **kwargs)
        self.default_bg = bg
        self.hover_bg = hover_bg
        self.command = command
        
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _on_enter(self, e):
        if str(self.cget("state")) != "disabled":
            self.config(bg=self.hover_bg)

    def _on_leave(self, e):
        if str(self.cget("state")) != "disabled":
            self.config(bg=self.default_bg)

    def _on_click(self, e):
        if str(self.cget("state")) != "disabled":
            self.command()

    def set_style(self, bg, fg, border=None):
        self.default_bg = bg
        self.config(bg=bg, fg=fg)
        if border:
            self.config(highlightbackground=border)


class ESRGApp:
    def __init__(self, root):
        self.root = root
        root.title("Enhanced SRG in MRI Image Segmentation")
        root.configure(bg=APP_BG)
        root.geometry("1380x860")
        root.minsize(1080, 680)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TScrollbar", gripcount=0, background=PANEL_ALT,
                        troughcolor=PANEL_BG, bordercolor=BORDER_CLR, arrowcolor=TEXT_MUTED)

        self.cfg = Config()
        self.image_path = None
        self.result = None
        self.current = "final"
        self.manual_points = []
        self._photo = None
        self.raw_image = None  # grayscale float64 preview shown before the pipeline runs
        self.norm_image = None  # pre.normalize(raw_image): what the pipeline grows on
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_start = None
        self._left_press = None
        self._left_dragging = False
        self.compare_overlay = tk.BooleanVar(value=False)
        self._compare_photos = []
        self.diag_visible = tk.BooleanVar(value=True)

        self.opacity = tk.DoubleVar(value=0.55)
        self.error_mode = tk.BooleanVar(value=False)
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

        self._build()

    # ── Master Layout ────────────────────────────────────────────────────────
    def _build(self):
        header = tk.Frame(self.root, bg=PANEL_BG, padx=18, pady=10,
                          highlightthickness=1, highlightbackground=BORDER_CLR)
        header.pack(side=tk.TOP, fill=tk.X)

        title_box = tk.Frame(header, bg=PANEL_BG)
        title_box.pack(side=tk.LEFT)
        tk.Label(title_box, text="ENHANCED SRG IN MRI IMAGE SEGMENTATION", bg=PANEL_BG,
                 fg=TEXT_MAIN, font=FONT_TITLE).pack(side=tk.LEFT)

        tk.Label(header, text="Enhanced Adams & Bischof (1994) Seeded Region Growing",
                 bg=PANEL_BG, fg=TEXT_FAINT, font=FONT_SM).pack(side=tk.RIGHT)

        score_strip = tk.Frame(self.root, bg=PANEL_BG, padx=16, pady=8,
                               highlightthickness=1, highlightbackground=BORDER_CLR)
        score_strip.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(0, 10))

        # Zoom controls live in the footer (not the stage/compare strip above
        # the viewer) so they stay put no matter which view is active.
        zoom_box = tk.Frame(score_strip, bg=PANEL_BG)
        zoom_box.pack(side=tk.RIGHT, padx=(8, 0))

        FlatButton(zoom_box, text="Reset", command=self._zoom_reset,
                   bg=PANEL_ALT, fg=TEXT_MUTED, hover_bg=BORDER_CLR,
                   font=FONT_SM, padx=8, pady=3).pack(side=tk.RIGHT, padx=(4, 0))
        FlatButton(zoom_box, text="+", command=self._zoom_in,
                   bg=PANEL_ALT, fg=TEXT_MAIN, hover_bg=BORDER_CLR,
                   font=("Segoe UI", 10, "bold"), padx=10, pady=3).pack(side=tk.RIGHT, padx=(4, 0))
        self.zoom_lbl = tk.Label(zoom_box, text="100%", bg=PANEL_BG, fg=TEXT_MUTED,
                                 font=FONT_SM, width=5, anchor="center")
        self.zoom_lbl.pack(side=tk.RIGHT, padx=(4, 0))
        FlatButton(zoom_box, text="−", command=self._zoom_out,
                   bg=PANEL_ALT, fg=TEXT_MAIN, hover_bg=BORDER_CLR,
                   font=("Segoe UI", 10, "bold"), padx=10, pady=3).pack(side=tk.RIGHT)
        tk.Label(zoom_box, text="ZOOM:", bg=PANEL_BG, fg=TEXT_FAINT,
                 font=FONT_BOLD).pack(side=tk.RIGHT, padx=(0, 6))

        # Lets the Stage Telemetry panel be hidden to reclaim width for the
        # viewer (e.g. the side-by-side Compare view) on narrower windows.
        self._diag_toggle_btn = FlatButton(
            score_strip, text="Telemetry ◂", command=self._toggle_diagnostics,
            bg=PANEL_ALT, fg=TEXT_MUTED, hover_bg=BORDER_CLR,
            font=FONT_SM, padx=8, pady=3)
        self._diag_toggle_btn.pack(side=tk.RIGHT, padx=(8, 0))

        tk.Label(score_strip, text="EVALUATION METRICS:", bg=PANEL_BG,
                 fg=PRIMARY, font=FONT_BOLD).pack(side=tk.LEFT, padx=(0, 8))
        self.score_lbl = tk.Label(score_strip, text="Load an MRI slice and run the pipeline to view metrics.",
                                  bg=PANEL_BG, fg=TEXT_MUTED, font=FONT_UI, anchor="w")
        self.score_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        body = tk.Frame(self.root, bg=APP_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        sidebar_frame = tk.Frame(body, bg=PANEL_BG, width=280,
                                 highlightthickness=1, highlightbackground=BORDER_CLR)
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar_frame.pack_propagate(False)
        self._build_sidebar(sidebar_frame)

        center_frame = tk.Frame(body, bg=APP_BG)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_viewer(center_frame)

        self.diagnostics_frame = tk.Frame(body, bg=PANEL_BG, width=320,
                                          highlightthickness=1, highlightbackground=BORDER_CLR)
        self.diagnostics_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        self.diagnostics_frame.pack_propagate(False)
        self._build_diagnostics(self.diagnostics_frame)

    # ── Sidebar & Parameter Controls ─────────────────────────────────────────
    def _section_header(self, parent, text):
        hdr = tk.Frame(parent, bg=PANEL_BG)
        hdr.pack(fill=tk.X, pady=(12, 4))
        tk.Label(hdr, text=text.upper(), bg=PANEL_BG, fg=PRIMARY,
                 font=FONT_BOLD, anchor="w").pack(side=tk.LEFT)
        tk.Frame(hdr, bg=BORDER_CLR, height=1).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

    def _build_sidebar(self, p):
        # 1. Pinned Action Buttons (Always visible at the bottom)
        action_frame = tk.Frame(p, bg=PANEL_BG, padx=12)
        action_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        tk.Frame(action_frame, bg=BORDER_CLR, height=1).pack(fill=tk.X, pady=(0, 10))

        self.run_btn = FlatButton(action_frame, text="RUN", command=self._run,
                                  bg=PRIMARY, fg="#ffffff", hover_bg=PRIMARY_HOV,
                                  border_color=PRIMARY, font=("Segoe UI", 10, "bold"), pady=8)
        self.run_btn.pack(fill=tk.X)

        self.status_lbl = tk.Label(action_frame, text="Ready", bg=PANEL_BG, fg=TEXT_MUTED,
                                   font=FONT_SM, anchor="w", wraplength=235, justify=tk.LEFT)
        self.status_lbl.pack(fill=tk.X, pady=(4, 8))

        # 2. Scrollable Parameters (Takes remaining top space)
        canvas = tk.Canvas(p, bg=PANEL_BG, highlightthickness=0)
        sb = ttk.Scrollbar(p, orient="vertical", command=canvas.yview)
        
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scroll_content = tk.Frame(canvas, bg=PANEL_BG, padx=12, pady=8)

        scroll_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        cw = canvas.create_window((0, 0), window=scroll_content, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        p.bind("<Configure>", lambda e: canvas.itemconfig(cw, width=e.width - 16))

        # Widgets embedded in a Canvas via create_window (like scroll_content
        # and everything in it) don't reliably receive real hardware
        # <MouseWheel> events routed to descendant widgets on macOS/Aqua,
        # even though per-widget bindings are technically present. Binding
        # globally via bind_all and gating on cursor position sidesteps that
        # routing entirely, so hovering anywhere over the sidebar scrolls it.
        def _on_sidebar_wheel(event):
            px, py = self.root.winfo_pointerx(), self.root.winfo_pointery()
            bx, by = p.winfo_rootx(), p.winfo_rooty()
            bw, bh = p.winfo_width(), p.winfo_height()
            if bw <= 1 or bh <= 1:
                return
            if not (bx <= px < bx + bw and by <= py < by + bh):
                return
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        self.root.bind_all("<MouseWheel>", _on_sidebar_wheel, add="+")
        self.root.bind_all("<Button-4>", _on_sidebar_wheel, add="+")
        self.root.bind_all("<Button-5>", _on_sidebar_wheel, add="+")

        # Input Source
        self._section_header(scroll_content, "Input Data")
        FlatButton(scroll_content, text="Open MRI Slice…", command=self._load,
                   bg=PANEL_ALT, fg=TEXT_MAIN, hover_bg=BORDER_CLR, font=FONT_UI, pady=6).pack(fill=tk.X, pady=(4, 2))

        self.file_lbl = tk.Label(scroll_content, text="No MRI scan selected", bg=PANEL_BG,
                                 fg=TEXT_FAINT, font=FONT_SM, anchor="w", wraplength=235, justify=tk.LEFT)
        self.file_lbl.pack(fill=tk.X, pady=(2, 6))

        # Pipeline Method
        self._section_header(scroll_content, "Pipeline Method")
        for val, lab in (("esrg", "ESRG (Enhanced Model)"), ("srg", "SRG Baseline (1994)")):
            tk.Radiobutton(scroll_content, text=lab, variable=self.method, value=val,
                           command=self._update_seed_region_visibility,
                           bg=PANEL_BG, fg=TEXT_MAIN, selectcolor=PANEL_ALT,
                           activebackground=PANEL_BG, font=FONT_UI, anchor="w",
                           highlightthickness=0).pack(fill=tk.X, pady=1)

        # Seeding Protocol
        self._section_header(scroll_content, "Seeding Strategy")
        for val, lab in (("auto", "Automated Seeding"), ("manual", "Manual Seeding")):
            tk.Radiobutton(scroll_content, text=lab, variable=self.seed_mode, value=val,
                           command=self._on_seed_mode_change,
                           bg=PANEL_BG, fg=TEXT_MAIN, selectcolor=PANEL_ALT,
                           activebackground=PANEL_BG, font=FONT_UI, anchor="w",
                           highlightthickness=0).pack(fill=tk.X, pady=1)

        # Seed region palette: clicks are planted as the selected region. It
        # only matters for Manual Landmark seeding and/or the SRG baseline,
        # which tessellates the head between all planted regions (region 2+
        # is what stops the tumor region from swallowing the whole slice) —
        # hidden otherwise by _update_seed_region_visibility.
        self.region_box = tk.Frame(scroll_content, bg=PANEL_BG)
        tk.Label(self.region_box, text="Seed region to plant", bg=PANEL_BG, fg=TEXT_MUTED,
                 font=FONT_SM, anchor="w").pack(fill=tk.X)
        for t in range(1, self.cfg.manual_seed_types + 1):
            row = tk.Frame(self.region_box, bg=PANEL_BG)
            row.pack(fill=tk.X)
            tk.Radiobutton(row, text=f"Region {t}" + (" (tumor)" if t == 1 else ""),
                           variable=self.seed_type, value=t, bg=PANEL_BG, fg=TEXT_MAIN,
                           selectcolor=PANEL_ALT, activebackground=PANEL_BG,
                           font=FONT_UI, anchor="w", highlightthickness=0).pack(side=tk.LEFT)
            tk.Frame(row, bg="#%02x%02x%02x" % SEED_RGB[(t - 1) % len(SEED_RGB)],
                     width=14, height=14).pack(side=tk.RIGHT, padx=6, pady=3)

        self._clear_seeds_btn = FlatButton(
            scroll_content, text="Clear Manual Seeds", command=self._clear_seeds,
            bg=PANEL_BG, fg=TEXT_MUTED, hover_bg=PANEL_ALT, font=FONT_SM, pady=4)
        self._clear_seeds_btn.pack(fill=tk.X, pady=(6, 4))
        self._update_seed_region_visibility()

        # Ablation Switches
        self._section_header(scroll_content, "Ablation Controls")
        for var, lab in ((self.use_log, "Log-Domain Transform (Obj 2)"),
                         (self.use_local, "Local Log Measure (Obj 2)"),
                         (self.use_stop, "Adaptive Termination (Obj 3)"),
                         (self.purify_manual_seed, "Purify Manual Seed (ESRG only)")):
            tk.Checkbutton(scroll_content, text=lab, variable=var, bg=PANEL_BG,
                           fg=TEXT_MAIN, selectcolor=PANEL_ALT, activebackground=PANEL_BG,
                           font=FONT_UI, anchor="w", highlightthickness=0).pack(fill=tk.X, pady=1)

        # Hyperparameters
        self._section_header(scroll_content, "Hyperparameters")
        self._slider(scroll_content, "k_L — Local Stopping Factor", self.k_local, 0.5, 4.0, 0.1)
        self._slider(scroll_content, "r — Neighborhood Radius", self.radius, 1, 8, 1)
        self._slider(scroll_content, "K — Otsu Threshold Classes", self.classes, 2, 5, 1)

        # Visualization Options
        self._section_header(scroll_content, "Display Settings")
        self._slider(scroll_content, "Overlay Opacity", self.opacity, 0.0, 1.0, 0.05, self._redraw)
        tk.Checkbutton(scroll_content, text="Show Error Heatmap (TP/FP/FN)",
                       variable=self.error_mode, bg=PANEL_BG, fg=TEXT_MAIN,
                       selectcolor=PANEL_ALT, activebackground=PANEL_BG,
                       font=FONT_UI, anchor="w", highlightthickness=0,
                       command=self._redraw).pack(fill=tk.X, pady=(2, 6))

    def _on_seed_mode_change(self):
        self._update_seed_region_visibility()
        self._refresh_telemetry()

    def _update_seed_region_visibility(self):
        """The seed-region palette only matters for Manual Landmark seeding;
        automatic SRG plants its own background seeds."""
        show = self.seed_mode.get() == "manual"
        if show:
            if not self.region_box.winfo_ismapped():
                self.region_box.pack(fill=tk.X, pady=(6, 2), before=self._clear_seeds_btn)
        else:
            self.region_box.pack_forget()

    def _slider(self, p, label, var, lo, hi, res, cmd=None):
        box = tk.Frame(p, bg=PANEL_BG)
        box.pack(fill=tk.X, pady=(2, 4))
        tk.Label(box, text=label, bg=PANEL_BG, fg=TEXT_MUTED,
                 font=FONT_SM, anchor="w").pack(fill=tk.X)
        tk.Scale(box, variable=var, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                 bg=PANEL_BG, fg=TEXT_MAIN, troughcolor=PANEL_ALT, activebackground=PRIMARY,
                 highlightthickness=0, bd=1, relief=tk.FLAT, font=FONT_SM,
                 command=(lambda _: cmd()) if cmd else None).pack(fill=tk.X)

    # ── Viewer & Stage Tab Strip ─────────────────────────────────────────────
    def _build_viewer(self, p):
        stage_strip_card = tk.Frame(p, bg=PANEL_BG, padx=8, pady=6,
                                    highlightthickness=1, highlightbackground=BORDER_CLR)
        stage_strip_card.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))

        tk.Label(stage_strip_card, text="STAGE:", bg=PANEL_BG, fg=TEXT_FAINT,
                 font=FONT_BOLD).pack(side=tk.LEFT, padx=(4, 8))

        # The tab row is horizontally scrollable rather than a plain Frame:
        # a plain Frame's natural width (sum of every stage tab + Compare)
        # otherwise inflates stage_strip_card's requested width, which in
        # turn inflates center_frame's requested width and can make Tk's
        # pack manager silently evict the diagnostics panel on the right for
        # lack of cavity space once enough tabs exist. A Canvas's requested
        # size is independent of its scrollable content, so this decouples
        # tab count from the rest of the window's layout entirely (same
        # pattern already used for the sidebar's vertical parameter scroll).
        tab_canvas = tk.Canvas(stage_strip_card, bg=PANEL_BG, height=28, highlightthickness=0)
        tab_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.stage_bar = tk.Frame(tab_canvas, bg=PANEL_BG)
        self.stage_buttons = {}
        tab_window = tab_canvas.create_window((0, 0), window=self.stage_bar, anchor="nw")
        self.stage_bar.bind(
            "<Configure>",
            lambda e: tab_canvas.configure(scrollregion=tab_canvas.bbox("all")))
        tab_canvas.bind(
            "<Configure>",
            lambda e: tab_canvas.itemconfig(tab_window, height=e.height))

        def _tab_scroll(event):
            if getattr(event, "num", None) == 4:
                tab_canvas.xview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                tab_canvas.xview_scroll(1, "units")
            else:
                tab_canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")

        tab_canvas.bind("<MouseWheel>", _tab_scroll)
        tab_canvas.bind("<Button-4>", _tab_scroll)
        tab_canvas.bind("<Button-5>", _tab_scroll)
        self._tab_scroll = _tab_scroll

        self._init_empty_stage_tabs()

        # Only meaningful while the COMPARE tab is active, but always visible
        # so it's easy to find alongside the other viewer controls.
        tk.Checkbutton(stage_strip_card, text="Overlay", variable=self.compare_overlay,
                       command=self._redraw, bg=PANEL_BG, fg=TEXT_MAIN,
                       selectcolor=PANEL_ALT, activebackground=PANEL_BG,
                       font=FONT_UI, highlightthickness=0).pack(side=tk.RIGHT, padx=(8, 4))

        viewer_card = tk.Frame(p, bg=VIEWPORT_BG, highlightthickness=1, highlightbackground=BORDER_MED)
        viewer_card.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(viewer_card, bg=VIEWPORT_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.canvas.bind("<MouseWheel>", self._on_wheel_zoom)
        self.canvas.bind("<Button-4>", self._on_wheel_zoom)
        self.canvas.bind("<Button-5>", self._on_wheel_zoom)
        # macOS synthesizes Control+MouseWheel (and Control+Button-4/5 on
        # X11-style setups) for trackpad pinch gestures in non-native-gesture
        # Tk apps, so pinch-to-zoom needs an explicit binding of its own.
        self.canvas.bind("<Control-MouseWheel>", self._on_wheel_zoom)
        self.canvas.bind("<Control-Button-4>", self._on_wheel_zoom)
        self.canvas.bind("<Control-Button-5>", self._on_wheel_zoom)
        # Right-click drag (desktop mouse) or left-click-and-hold drag
        # (trackpad-friendly, no right-click gesture needed) both pan around
        # the zoomed-in image. A plain left click with no meaningful drag
        # still places a manual seed, via the click/drag disambiguation in
        # _on_left_press/_on_left_drag/_on_left_release.
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)

        # Side-by-side Compare pane: two independently-clipped canvases so a
        # zoomed-in mask can never bleed across into the other panel (a
        # single shared canvas only clips to its own outer bounds, not to
        # sub-regions drawn onto it). Built once, hidden until Compare mode
        # with "Overlay" unchecked is actually selected (see
        # _sync_viewer_visibility).
        self.compare_pane = tk.Frame(viewer_card, bg=VIEWPORT_BG)

        left_box = tk.Frame(self.compare_pane, bg=VIEWPORT_BG)
        left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(left_box, text="ESRG SEGMENTATION MASK", bg=VIEWPORT_BG,
                 fg="#f87171", font=FONT_BOLD).pack(side=tk.TOP, fill=tk.X, pady=(6, 4))
        self.cmp_canvas_l = tk.Canvas(left_box, bg=VIEWPORT_BG, highlightthickness=0)
        self.cmp_canvas_l.pack(fill=tk.BOTH, expand=True)

        tk.Frame(self.compare_pane, bg=BORDER_MED, width=2).pack(side=tk.LEFT, fill=tk.Y)

        right_box = tk.Frame(self.compare_pane, bg=VIEWPORT_BG)
        right_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(right_box, text="GROUND TRUTH MASK", bg=VIEWPORT_BG,
                 fg="#4ade80", font=FONT_BOLD).pack(side=tk.TOP, fill=tk.X, pady=(6, 4))
        self.cmp_canvas_r = tk.Canvas(right_box, bg=VIEWPORT_BG, highlightthickness=0)
        self.cmp_canvas_r.pack(fill=tk.BOTH, expand=True)

        for c in (self.cmp_canvas_l, self.cmp_canvas_r):
            c.bind("<Configure>", lambda e: self._redraw())
            c.bind("<MouseWheel>", self._on_wheel_zoom)
            c.bind("<Button-4>", self._on_wheel_zoom)
            c.bind("<Button-5>", self._on_wheel_zoom)
            c.bind("<Control-MouseWheel>", self._on_wheel_zoom)
            c.bind("<Control-Button-4>", self._on_wheel_zoom)
            c.bind("<Control-Button-5>", self._on_wheel_zoom)
            c.bind("<ButtonPress-3>", self._on_pan_start)
            c.bind("<B3-Motion>", self._on_pan_move)
            # No seed placement is meaningful on these panels, so plain
            # left-click-and-drag can pan immediately with no click/drag
            # disambiguation needed.
            c.bind("<ButtonPress-1>", self._on_pan_start)
            c.bind("<B1-Motion>", self._on_pan_move)

    def _init_empty_stage_tabs(self):
        self._clear_stages()
        lbl = tk.Label(self.stage_bar, text="No stages processed yet",
                       bg=PANEL_BG, fg=TEXT_FAINT, font=FONT_SM)
        lbl.pack(side=tk.LEFT, padx=4)
        lbl.bind("<MouseWheel>", self._tab_scroll)
        lbl.bind("<Button-4>", self._tab_scroll)
        lbl.bind("<Button-5>", self._tab_scroll)

    # ── Diagnostics & Telemetry Panel ────────────────────────────────────────
    def _build_diagnostics(self, p):
        head = tk.Frame(p, bg=PANEL_BG, padx=10, pady=8)
        head.pack(fill=tk.X)
        tk.Label(head, text="SEED TELEMETRY", bg=PANEL_BG, fg=PRIMARY,
                 font=FONT_BOLD).pack(side=tk.LEFT)

        container = tk.Frame(p, bg=PANEL_BG, padx=8)
        container.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        sb = ttk.Scrollbar(container)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.diag = tk.Text(container, bg=PANEL_ALT, fg=TEXT_MAIN, font=FONT_MONO,
                            wrap=tk.WORD, relief=tk.SOLID, bd=1, highlightthickness=0,
                            yscrollcommand=sb.set, padx=10, pady=10)
        self.diag.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.diag.yview)

        self.diag.tag_config("head", foreground=PRIMARY, font=FONT_MONO + ("bold",))
        self.diag.tag_config("muted", foreground=TEXT_MUTED)
        self.diag.tag_config("avg", foreground=TEXT_MAIN, font=FONT_MONO + ("bold",))
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

    def _refresh_telemetry(self):
        """Lists the coordinate and grayscale value of every seed pixel, with the
        mean of each region when it holds more than one pixel."""
        groups, msg = self._telemetry_groups()
        d = self.diag
        top = d.yview()[0]
        d.config(state=tk.NORMAL)
        d.delete("1.0", tk.END)

        if msg:
            d.insert(tk.END, msg + "\n", "muted")
        elif self.result and self.current == "log":
            self._write_log_telemetry(groups)
        else:
            h, w = self.raw_image.shape
            mode = "Manual" if self.seed_mode.get() == "manual" else "Automated"
            d.insert(tk.END, f"{w} × {h} px · {mode} seeding\n", "head")
            d.insert(tk.END, "row = y, col = x (0-based)\n"
                             "RAW = file value\n"
                             "NORM = value the algorithm uses (0–255)\n", "muted")
            for g in pixel_report.describe(groups, self.raw_image, self.norm_image):
                d.insert(tk.END, "\n" + "─" * 34 + "\n")
                d.insert(tk.END, f"{g['label']} — {g['n']} px\n", "head")
                rows = [f"{'row':>5}{'col':>6}{'raw':>7}{'norm':>9}"]
                rows += [f"{p['row']:>5}{p['col']:>6}{p['raw']:>7.0f}{p['norm']:>9.2f}"
                         for p in g["pixels"]]
                d.insert(tk.END, "\n".join(rows) + "\n")
                if g["n"] > 1:
                    d.insert(tk.END, f"Average  raw {g['mean_raw']:.2f}  norm {g['mean_norm']:.2f}\n", "avg")

        d.yview_moveto(top)
        d.config(state=tk.DISABLED)

    def _write_log_telemetry(self, groups):
        """Stage 3 view: L(x) of every seed pixel with its arithmetic, and the
        noise floor's intermediates, all read from the arrays the run used."""
        d = self.diag
        st = self.result.stage("log")
        mask = self.result.stage("mask").image
        I = np.where(mask, self.result.stage("input").image, 0.0)
        use_log, eps = st.info["_use_log"], st.info["_eps"]
        nf = st.info["_noise_floor"]

        d.insert(tk.END, "3 · Log domain\n", "head")
        if use_log:
            d.insert(tk.END, f"L(x) = ln(I(x) + ε),  ε = {eps:g}\n", "avg")
            d.insert(tk.END, "ln = natural log (base e)\n"
                             "I = NORM value inside head mask H\n"
                             "    (0 outside H)\n", "muted")
        else:
            d.insert(tk.END, "Log transform OFF: L(x) = I(x)\n", "avg")
            d.insert(tk.END, "I = NORM value inside head mask H\n"
                             "Pixels pass through unchanged.\n", "muted")

        for g in pixel_report.log_domain(groups, I, st.image, mask, eps, use_log):
            d.insert(tk.END, "\n" + "─" * 34 + "\n")
            d.insert(tk.END, f"{g['label']} — {g['n']} px\n", "head")
            d.insert(tk.END, f"{'row':>5}{'col':>6}{'I(x)':>12}{'L(x)':>11}\n")
            for p in g["pixels"]:
                d.insert(tk.END, f"{p['row']:>5}{p['col']:>6}{p['I']:>12.6f}{p['L']:>11.6f}\n")
                if not p["in_head"]:
                    d.insert(tk.END, "  outside H, so I(x) = 0\n", "muted")
                if use_log:
                    d.insert(tk.END, f"  ln({p['I']:.6f} + {eps:g}) = {p['L']:.6f}\n", "muted")
            if g["n"] > 1:
                d.insert(tk.END, f"Average  L {g['mean_L']:.6f}\n", "avg")

        d.insert(tk.END, "\n" + "─" * 34 + "\n")
        d.insert(tk.END, "Noise floor σ_floor\n", "head")
        d.insert(tk.END, f"over all {nf['n']} px of H\n"
                         "r(x) = L(x) − median3×3(L)(x)\n"
                         "MAD  = median|r − median(r)|\n", "muted")
        d.insert(tk.END, f"median(r)     = {nf['median_resid']:.6f}\n"
                         f"MAD           = {nf['mad']:.6f}\n"
                         f"1.4826 × MAD  = {nf['raw_sigma']:.6f}\n")
        d.insert(tk.END, f"σ_floor = max({nf['raw_sigma']:.6f}, {nf['min_value']:g})\n"
                         f"        = {nf['sigma_floor']:.6f}\n", "avg")

    def _toggle_diagnostics(self):
        self.diag_visible.set(not self.diag_visible.get())
        if self.diag_visible.get():
            self.diagnostics_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
            self._diag_toggle_btn.config(text="Telemetry ◂")
        else:
            self.diagnostics_frame.pack_forget()
            self._diag_toggle_btn.config(text="Telemetry ▸")
        # Force Tk to recompute geometry synchronously so the redraw below
        # already sees the post-toggle canvas size instead of a stale one
        # (otherwise it only self-corrects on the next natural <Configure>).
        self.root.update_idletasks()
        self._redraw()

    # ── Controller & Backend Invocation ──────────────────────────────────────
    def _load(self):
        path = filedialog.askopenfilename(
            title="Select MRI Slice",
            filetypes=[("Medical Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All Files", "*.*")])
        if not path:
            return
        self.image_path = path
        self.result = None
        self.manual_points = []

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
        self._refresh_telemetry()

        gt = find_mask_for(path)
        gt_status = "Ground truth detected" if gt else "No ground truth found"
        self.file_lbl.config(text=f"{os.path.basename(path)}\n• {gt_status}", fg=TEXT_MUTED)
        self._init_empty_stage_tabs()
        self.score_lbl.config(text="MRI slice loaded — verify the preview, then press 'RUN'.", fg=TEXT_MAIN)
        self._redraw()

    def _clear_seeds(self):
        self.manual_points = []
        self.status_lbl.config(text="Manual seed coordinates cleared.", fg=TEXT_MUTED)
        self._refresh_telemetry()
        self._redraw()

    def _on_click(self, event):
        if self.seed_mode.get() != "manual" or not self.image_path:
            return
        rc = self._canvas_to_image(event.x, event.y)
        if rc:
            t = self.seed_type.get()
            self.manual_points.append((rc[0], rc[1], t))
            n_regions = len({p[2] for p in self.manual_points})
            self.status_lbl.config(
                text=f"{len(self.manual_points)} seed(s) across {n_regions} region(s) placed.",
                fg=PRIMARY)
            self._refresh_telemetry()
            self._redraw()

    def _current_config(self):
        return self.cfg.replace(
            method=self.method.get(), seed_mode=self.seed_mode.get(),
            use_log_local=self.use_local.get(), use_stopping=self.use_stop.get(),
            use_log=self.use_log.get(), purify_manual_seed=self.purify_manual_seed.get(),
            k_local=float(self.k_local.get()),
            k_global=max(float(self.k_local.get()) + 1.0, 3.0),
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
        self.run_btn.config(state="disabled", text="Processing…", cursor="arrow")
        self.run_btn.set_style(bg=BORDER_MED, fg=TEXT_MAIN, border=BORDER_MED)
        self.status_lbl.config(text="Segmenting slice…", fg=PRIMARY)

        def worker():
            try:
                res = run(self.image_path, cfg, manual_points=self.manual_points,
                          progress=lambda m: self.root.after(0, self.status_lbl.config, {"text": m}))
                self.root.after(0, self._done, res, None)
            except Exception as e:
                self.root.after(0, self._done, None, e)

        threading.Thread(target=worker, daemon=True).start()

    def _done(self, res, err):
        self.run_btn.config(state="normal", text="RUN", cursor="hand2")
        self.run_btn.set_style(bg=PRIMARY, fg="#ffffff", border=PRIMARY)
        
        if err:
            self.status_lbl.config(text="Pipeline execution failed.", fg=FAIL_CLR)
            messagebox.showerror("Execution Error", str(err))
            return

        self.result = res
        self.status_lbl.config(text=f"Status: {res.status}", fg=OK_CLR if res.status == "OK" else WARN_CLR)
        self._build_stage_buttons()
        self.current = "final"
        self._select("final")
        self._refresh_telemetry()

        if res.status != "NO TUMOR CANDIDATE":
            self.score_lbl.config(text=viz.score_line(res.scores), fg=TEXT_MAIN)
        else:
            self.score_lbl.config(
                text="NO TUMOR CANDIDATE DETECTED — Seeds were filtered during interior checks.",
                fg=FAIL_CLR
            )

    def _clear_stages(self):
        for w in self.stage_bar.winfo_children():
            w.destroy()
        self.stage_buttons = {}

    def _build_stage_buttons(self):
        self._clear_stages()
        status_colors = {"OK": OK_CLR, "WARN": WARN_CLR, "FAIL": FAIL_CLR}

        for st in self.result.stages:
            badge_color = status_colors.get(st.status, TEXT_MUTED)

            btn = FlatButton(
                self.stage_bar,
                text=f"{st.name} ●",
                command=lambda k=st.key: self._select(k),
                bg=PANEL_ALT,
                fg=badge_color,
                hover_bg=BORDER_CLR,
                font=FONT_BOLD,
                padx=10,
                pady=4
            )
            btn.pack(side=tk.LEFT, padx=(0, 4))
            btn.bind("<MouseWheel>", self._tab_scroll, add="+")
            btn.bind("<Button-4>", self._tab_scroll, add="+")
            btn.bind("<Button-5>", self._tab_scroll, add="+")
            self.stage_buttons[st.key] = btn

        if self.result.gt is not None:
            btn = FlatButton(
                self.stage_bar,
                text="Compare ●",
                command=lambda: self._select("compare"),
                bg=PANEL_ALT,
                fg=TEXT_MUTED,
                hover_bg=BORDER_CLR,
                font=FONT_BOLD,
                padx=10,
                pady=4
            )
            btn.pack(side=tk.LEFT, padx=(0, 4))
            btn.bind("<MouseWheel>", self._tab_scroll, add="+")
            btn.bind("<Button-4>", self._tab_scroll, add="+")
            btn.bind("<Button-5>", self._tab_scroll, add="+")
            self.stage_buttons["compare"] = btn

    def _select(self, key):
        self.current = key
        for k, b in self.stage_buttons.items():
            if k == key:
                b.set_style(bg=PRIMARY, fg="#ffffff", border=PRIMARY)
            else:
                st = self.result.stage(k)
                status_colors = {"OK": OK_CLR, "WARN": WARN_CLR, "FAIL": FAIL_CLR}
                color = status_colors.get(st.status, TEXT_MUTED) if st else TEXT_MUTED
                b.set_style(bg=PANEL_ALT, fg=color, border=BORDER_CLR)

        self._refresh_telemetry()
        self._redraw()

    # ── Zoom & Pan ────────────────────────────────────────────────────────────
    def _has_image(self):
        return self.result is not None or self.raw_image is not None

    def _is_compare_active(self):
        return (self.result is not None and self.current == "compare"
                and self.result.gt is not None)

    def _wants_split_compare(self):
        return self._is_compare_active() and not self.compare_overlay.get()

    def _set_zoom(self, z):
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, z))
        self.zoom_lbl.config(text=f"{round(self.zoom * 100)}%")
        self._redraw()

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
        # that same point stays under the cursor afterwards. Both compare
        # panels share identical w/h/s and only differ by a constant offset,
        # so solving in whichever panel the cursor is over and writing the
        # result into the shared pan keeps them moving in lockstep.
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
        self._redraw()

    # Left-click-and-hold drag pans the main viewer too (a trackpad has no
    # natural right-click-drag gesture), while a plain click with no real
    # movement still places a manual seed. Distinguish the two by how far
    # the pointer actually moved before release.
    _DRAG_THRESHOLD = 4

    def _on_left_press(self, event):
        if self._has_image():
            self._left_press = (event.x, event.y, self.pan_x, self.pan_y)
            self._left_dragging = False

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
            self._redraw()

    def _on_left_release(self, event):
        if self._left_press and not self._left_dragging:
            self._on_click(event)
        self._left_press = None
        self._left_dragging = False

    # ── Rendering & Visual Geometry ──────────────────────────────────────────
    def _sync_viewer_visibility(self):
        """Show self.canvas or the split self.compare_pane depending on the
        active view, so exactly one of them is packed at any time. Each
        canvas naturally clips its own drawing to its own bounds, which is
        what keeps zoomed compare panels from bleeding into one another."""
        want_split = self._wants_split_compare()
        is_split = bool(self.compare_pane.winfo_ismapped())
        if want_split == is_split:
            return
        if want_split:
            self.canvas.pack_forget()
            self.compare_pane.pack(fill=tk.BOTH, expand=True)
        else:
            self.compare_pane.pack_forget()
            self.canvas.pack(fill=tk.BOTH, expand=True)
        # The canvas layout class just changed (one full-width canvas <->
        # two half-width canvases) so a raw pixel pan offset from the old
        # layout is meaningless in the new one. Zoom level is kept.
        self.pan_x = 0.0
        self.pan_y = 0.0

    def _geometry(self, canvas=None):
        canvas = canvas or self.canvas
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

    def _canvas_to_image(self, x, y):
        g = self._geometry()
        if not g:
            return None
        w, h, s, ox, oy = g
        c, r = int((x - ox) / s), int((y - oy) / s)
        return (r, c) if 0 <= r < h and 0 <= c < w else None

    def _redraw(self):
        self._sync_viewer_visibility()

        if self._wants_split_compare():
            self._redraw_compare_split()
            return

        self.canvas.delete("all")
        cw = max(self.canvas.winfo_width(), 50)
        ch = max(self.canvas.winfo_height(), 50)

        if not self.result and self.raw_image is None:
            self.canvas.create_text(
                cw // 2, ch // 2,
                text="Load an MRI slice to initialize viewport\n(Manual landmark placement is active in Manual mode)\n"
                     "Scroll or use +/− to zoom, right-click drag to pan",
                fill=TEXT_FAINT,
                font=FONT_SUB,
                justify=tk.CENTER
            )
            return

        if not self.result:
            # Raw scan preview, shown before the pipeline has run so the user
            # can confirm the correct slice was loaded.
            rgb = np.repeat(self.raw_image[:, :, None], 3, axis=2)
        elif self._is_compare_active():
            self._redraw_compare_overlay(cw, ch)
            return
        else:
            st = self.result.stage(self.current) or self.result.stages[-1]
            base = self.result.stage("input").image

            if st.key == "final":
                rgb = viz.overlay_result(base, st.image, self.result.gt,
                                         self.opacity.get(), self.error_mode.get())
            else:
                rgb = viz.render_stage(st, base, self.result.gt, self.opacity.get())

        if self.seed_mode.get() == "manual" and self.manual_points:
            for r, c, t in self.manual_points:
                rgb[max(0, r - 2):r + 3, max(0, c - 2):c + 3] = SEED_RGB[(t - 1) % len(SEED_RGB)]

        w, h, s, ox, oy = self._geometry()
        img = Image.fromarray(rgb.astype(np.uint8)).resize(
            (max(1, int(w * s)), max(1, int(h * s))), Image.NEAREST
        )
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self._photo)

    @staticmethod
    def _mask_rgb(base_img, mask, color, opacity):
        rgb = np.stack([np.clip(base_img, 0, 255).astype(np.uint8)] * 3, axis=-1)
        m = mask.astype(bool)
        if m.any():
            rgb[m] = (np.array(color) * opacity + rgb[m] * (1 - opacity)).astype(np.uint8)
        return rgb

    def _redraw_compare_overlay(self, cw, ch):
        """Blended overlay of both masks on one canvas (Overlay toggle checked)."""
        base = self.result.stage("input").image
        pred = self.result.mask.astype(bool)
        gt = self.result.gt.astype(bool)
        h, w = base.shape

        rgb = viz.overlay_result(base, pred, gt, self.opacity.get(), self.error_mode.get())
        s = min(cw / w, ch / h) * self.zoom
        iw, ih = max(1, int(w * s)), max(1, int(h * s))
        img = Image.fromarray(rgb.astype(np.uint8)).resize((iw, ih), Image.NEAREST)
        photo = ImageTk.PhotoImage(img)
        self._compare_photos = [photo]
        ox, oy = (cw - iw) / 2 + self.pan_x, (ch - ih) / 2 + self.pan_y
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=photo)
        label = ("OVERLAY — TP green / FP red / FN orange" if self.error_mode.get()
                 else "OVERLAY — ESRG prediction (red fill) vs ground truth (green outline)")
        self.canvas.create_text(cw / 2, 14, text=label, fill=TEXT_FAINT, font=FONT_SM)

    def _redraw_compare_split(self):
        """Side-by-side ESRG prediction vs. ground truth, each on its own
        canvas so a zoomed-in mask is hard-clipped to its own panel and can
        never bleed into the other one."""
        base = self.result.stage("input").image
        pred = self.result.mask.astype(bool)
        gt = self.result.gt.astype(bool)

        left_rgb = self._mask_rgb(base, pred, viz.RED, self.opacity.get())
        right_rgb = self._mask_rgb(base, gt, viz.GREEN, self.opacity.get())

        self._compare_photos = []
        for canvas, rgb in ((self.cmp_canvas_l, left_rgb), (self.cmp_canvas_r, right_rgb)):
            canvas.delete("all")
            g = self._geometry(canvas)
            if not g:
                continue
            w, h, s, ox, oy = g
            iw, ih = max(1, int(w * s)), max(1, int(h * s))
            img = Image.fromarray(rgb.astype(np.uint8)).resize((iw, ih), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            self._compare_photos.append(photo)
            canvas.create_image(ox, oy, anchor=tk.NW, image=photo)


if __name__ == "__main__":
    root = tk.Tk()
    ESRGApp(root)
    root.mainloop()