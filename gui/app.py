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

# Manual seed types share the tessellation palette (esrg.visualize.SEED_COLORS):
# type 1 is whichever region the user treats as the structure of interest,
# type 2+ are the other competing regions. SRG itself does not distinguish them.
SEED_RGB = viz.SEED_COLORS

# Typography
FONT_TITLE  = ("Segoe UI", 11, "bold")
FONT_SUB    = ("Segoe UI", 9)
FONT_BOLD   = ("Segoe UI", 8, "bold")
FONT_UI     = ("Segoe UI", 9)
FONT_SM     = ("Segoe UI", 8)
FONT_MONO   = ("Menlo", 8)


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
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_start = None

        self.opacity = tk.DoubleVar(value=0.55)
        self.error_mode = tk.BooleanVar(value=False)
        self.method = tk.StringVar(value="esrg")
        self.seed_mode = tk.StringVar(value="auto")
        self.seed_type = tk.IntVar(value=1)
        self.use_local = tk.BooleanVar(value=True)
        self.use_stop = tk.BooleanVar(value=True)
        self.use_n4 = tk.BooleanVar(value=False)
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

        diagnostics_frame = tk.Frame(body, bg=PANEL_BG, width=320,
                                     highlightthickness=1, highlightbackground=BORDER_CLR)
        diagnostics_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        diagnostics_frame.pack_propagate(False)
        self._build_diagnostics(diagnostics_frame)

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

        self.run_btn = FlatButton(action_frame, text="RUN PIPELINE", command=self._run,
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

        def _on_mousewheel(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        # scroll_content fully covers the canvas, so hovering anywhere over
        # the panel actually targets one of its child widgets (a label,
        # radiobutton, frame, ...), not the canvas itself. Bind the wheel
        # directly to every one of them, walking the tree as it's built.
        def _bind_wheel_tree(widget):
            widget.bind("<MouseWheel>", _on_mousewheel, add="+")
            widget.bind("<Button-4>", _on_mousewheel, add="+")
            widget.bind("<Button-5>", _on_mousewheel, add="+")
            for child in widget.winfo_children():
                _bind_wheel_tree(child)

        self._sidebar_bind_wheel = _bind_wheel_tree
        _bind_wheel_tree(canvas)

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
                           bg=PANEL_BG, fg=TEXT_MAIN, selectcolor=PANEL_ALT,
                           activebackground=PANEL_BG, font=FONT_UI, anchor="w",
                           highlightthickness=0).pack(fill=tk.X, pady=1)

        # Seeding Protocol
        self._section_header(scroll_content, "Seeding Strategy")
        for val, lab in (("auto", "Automated Candidate (Obj 1)"), ("manual", "Manual Landmark Seed")):
            tk.Radiobutton(scroll_content, text=lab, variable=self.seed_mode, value=val,
                           bg=PANEL_BG, fg=TEXT_MAIN, selectcolor=PANEL_ALT,
                           activebackground=PANEL_BG, font=FONT_UI, anchor="w",
                           highlightthickness=0).pack(fill=tk.X, pady=1)

        # Seed type palette: clicks are planted as the selected type. The SRG
        # baseline tessellates the head between all planted types, so types 2+
        # are what stop the tumor region from swallowing the whole slice.
        type_box = tk.Frame(scroll_content, bg=PANEL_BG)
        type_box.pack(fill=tk.X, pady=(6, 2))
        tk.Label(type_box, text="Seed type to plant", bg=PANEL_BG, fg=TEXT_MUTED,
                 font=FONT_SM, anchor="w").pack(fill=tk.X)
        for t in range(1, self.cfg.manual_seed_types + 1):
            row = tk.Frame(type_box, bg=PANEL_BG)
            row.pack(fill=tk.X)
            tk.Radiobutton(row, text=f"Type {t}" + (" (tumor)" if t == 1 else ""),
                           variable=self.seed_type, value=t, bg=PANEL_BG, fg=TEXT_MAIN,
                           selectcolor=PANEL_ALT, activebackground=PANEL_BG,
                           font=FONT_UI, anchor="w", highlightthickness=0).pack(side=tk.LEFT)
            tk.Frame(row, bg="#%02x%02x%02x" % SEED_RGB[(t - 1) % len(SEED_RGB)],
                     width=14, height=14).pack(side=tk.RIGHT, padx=6, pady=3)

        FlatButton(scroll_content, text="Clear Manual Seeds", command=self._clear_seeds,
                   bg=PANEL_BG, fg=TEXT_MUTED, hover_bg=PANEL_ALT, font=FONT_SM, pady=4).pack(fill=tk.X, pady=(6, 4))

        # Ablation Switches
        self._section_header(scroll_content, "Ablation Controls")
        for var, lab in ((self.use_local, "Local Log Measure (Obj 2)"),
                         (self.use_stop, "Adaptive Termination (Obj 3)"),
                         (self.use_n4, "N4 Bias Correction")):
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

        self._sidebar_bind_wheel(scroll_content)

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

        self.stage_bar = tk.Frame(stage_strip_card, bg=PANEL_BG)
        self.stage_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.stage_buttons = {}

        self._init_empty_stage_tabs()

        zoom_box = tk.Frame(stage_strip_card, bg=PANEL_BG)
        zoom_box.pack(side=tk.RIGHT, padx=(8, 4))

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

        viewer_card = tk.Frame(p, bg=VIEWPORT_BG, highlightthickness=1, highlightbackground=BORDER_MED)
        viewer_card.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(viewer_card, bg=VIEWPORT_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.canvas.bind("<MouseWheel>", self._on_wheel_zoom)
        self.canvas.bind("<Button-4>", self._on_wheel_zoom)
        self.canvas.bind("<Button-5>", self._on_wheel_zoom)
        # Right-click drag pans around the zoomed-in image (left click is
        # reserved for placing manual seeds).
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)

    def _init_empty_stage_tabs(self):
        self._clear_stages()
        lbl = tk.Label(self.stage_bar, text="No stages processed yet",
                       bg=PANEL_BG, fg=TEXT_FAINT, font=FONT_SM)
        lbl.pack(side=tk.LEFT, padx=4)

    # ── Diagnostics & Telemetry Panel ────────────────────────────────────────
    def _build_diagnostics(self, p):
        head = tk.Frame(p, bg=PANEL_BG, padx=10, pady=8)
        head.pack(fill=tk.X)
        tk.Label(head, text="STAGE TELEMETRY", bg=PANEL_BG, fg=PRIMARY,
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

        self.diag.insert("1.0", "Execute the segmentation pipeline to inspect per-stage metrics and parameters.\n")
        self.diag.config(state=tk.DISABLED)

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
        except Exception as e:
            self.raw_image = None
            messagebox.showerror("Load Failed", f"Could not read this image:\n{e}")
            return

        self._zoom_reset()

        gt = find_mask_for(path)
        gt_status = "Ground truth detected" if gt else "No ground truth found"
        self.file_lbl.config(text=f"{os.path.basename(path)}\n• {gt_status}", fg=TEXT_MUTED)
        self._init_empty_stage_tabs()
        self.score_lbl.config(text="MRI slice loaded — verify the preview, then press 'RUN PIPELINE'.", fg=TEXT_MAIN)
        self._redraw()

    def _clear_seeds(self):
        self.manual_points = []
        self.status_lbl.config(text="Manual seed coordinates cleared.", fg=TEXT_MUTED)
        self._redraw()

    def _on_click(self, event):
        if self.seed_mode.get() != "manual" or not self.image_path:
            return
        rc = self._canvas_to_image(event.x, event.y)
        if rc:
            t = self.seed_type.get()
            self.manual_points.append((rc[0], rc[1], t))
            n_types = len({p[2] for p in self.manual_points})
            self.status_lbl.config(
                text=f"{len(self.manual_points)} seed(s) across {n_types} type(s) placed.",
                fg=PRIMARY)
            self._redraw()

    def _current_config(self):
        return self.cfg.replace(
            method=self.method.get(), seed_mode=self.seed_mode.get(),
            use_log_local=self.use_local.get(), use_stopping=self.use_stop.get(),
            use_n4=self.use_n4.get(), k_local=float(self.k_local.get()),
            k_global=max(float(self.k_local.get()) + 1.0, 3.0),
            local_radius=int(self.radius.get()), otsu_classes=int(self.classes.get()))

    def _run(self):
        if not self.image_path:
            messagebox.showwarning("Input Required", "Please load an MRI slice before running.")
            return
        if self.seed_mode.get() == "manual" and not any(p[2] == 1 for p in self.manual_points):
            messagebox.showwarning("Seed Required",
                                   "Click on the tumor region to place at least one Type 1 seed.")
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
        self.run_btn.config(state="normal", text="RUN PIPELINE", cursor="hand2")
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
            self.stage_buttons[st.key] = btn

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

        self._show_diagnostics(key)
        self._redraw()

    def _show_diagnostics(self, key):
        st = self.result.stage(key)
        self.diag.config(state=tk.NORMAL)
        self.diag.delete("1.0", tk.END)

        if st:
            header_str = f"STAGE: {st.name.upper()}\nStatus   : {st.status}\nDuration : {st.seconds * 1000:.1f} ms\n"
            self.diag.insert(tk.END, header_str)
            self.diag.insert(tk.END, "-" * 34 + "\n")

            for k, v in st.info.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    self.diag.insert(tk.END, f"\n[{k}]\n")
                    for item in v:
                        self.diag.insert(tk.END, "  • " + ", ".join(f"{a}: {b}" for a, b in item.items()) + "\n")
                else:
                    self.diag.insert(tk.END, f"{k:<18}: {v}\n")

            if key == "final" and self.result.scores:
                self.diag.insert(tk.END, "\n" + "=" * 34 + "\nSEGMENTATION ACCURACY:\n")
                for k, v in self.result.scores.items():
                    self.diag.insert(tk.END, f"  {k:<16}: {v}\n")

        self.diag.config(state=tk.DISABLED)

    # ── Zoom & Pan ────────────────────────────────────────────────────────────
    def _has_image(self):
        return self.result is not None or self.raw_image is not None

    def _set_zoom(self, z):
        self.zoom = max(0.25, min(8.0, z))
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
        if getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
            self._zoom_out()
        else:
            self._zoom_in()

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

    # ── Rendering & Visual Geometry ──────────────────────────────────────────
    def _geometry(self):
        if self.result:
            h, w = self.result.stage("input").image.shape
        elif self.raw_image is not None:
            h, w = self.raw_image.shape
        else:
            return None
        cw = max(self.canvas.winfo_width(), 50)
        ch = max(self.canvas.winfo_height(), 50)
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

if __name__ == "__main__":
    root = tk.Tk()
    ESRGApp(root)
    root.mainloop()