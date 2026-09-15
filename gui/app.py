"""
app.py — Desktop GUI for the ESRG pipeline.

Purpose : Load a slice, run the pipeline, inspect any stage, and read the scores —
          giving per-stage error tracking and an annotated output.
Function : A Tkinter app with a controls sidebar (method, seeding, ablation switches,
          parameters), a centre viewer with one button per pipeline stage, a
          diagnostics panel, and a score bar; runs the pipeline off the UI thread and
          can save every stage image plus the mask, overlay, and config.
Notes   : Manual seeding lets the user click the tumor for the hard cases (pituitary,
          glioma) where automatic seeding is unreliable. Run: python -m gui.app
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
from esrg.io_utils import find_mask_for

BG, PANEL, FG, DIM, ACCENT = "#1a1a2e", "#0f172a", "#e2e8f0", "#94a3b8", "#3b82f6"
OK, WARN, FAIL = "#22c55e", "#f59e0b", "#ef4444"
MONO = ("Courier", 9)


class ESRGApp:
    def __init__(self, root):
        self.root = root
        root.title("Enhanced Seeded Region Growing — Brain Tumor Segmentation")
        root.configure(bg=BG)
        root.geometry("1320x820")
        root.minsize(1000, 640)

        self.cfg = Config()
        self.image_path = None
        self.result = None
        self.current = "final"
        self.manual_points = []
        self._photo = None

        self.opacity = tk.DoubleVar(value=0.55)
        self.error_mode = tk.BooleanVar(value=False)
        self.method = tk.StringVar(value="esrg")
        self.seed_mode = tk.StringVar(value="auto")
        self.use_local = tk.BooleanVar(value=True)
        self.use_stop = tk.BooleanVar(value=True)
        self.use_n4 = tk.BooleanVar(value=False)
        self.k_local = tk.DoubleVar(value=self.cfg.k_local)
        self.radius = tk.IntVar(value=self.cfg.local_radius)
        self.classes = tk.IntVar(value=self.cfg.otsu_classes)

        self._build()

    # ── Layout ───────────────────────────────────────────────────────────────
    def _build(self):
        bar = tk.Frame(self.root, bg="#0f0f1a", pady=6, padx=12)
        bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(bar, text="ESRG  |  Brain Tumor Segmentation", bg="#0f0f1a",
                 fg=DIM, font=MONO).pack(side=tk.LEFT)
        tk.Label(bar, text="Enhancement of Adams & Bischof (1994)", bg="#0f0f1a",
                 fg="#334155", font=("Courier", 8)).pack(side=tk.RIGHT)

        self.score_lbl = tk.Label(self.root, text="Load an MRI slice to begin.",
                                  bg=PANEL, fg=FG, font=MONO, anchor="w", pady=8, padx=12)
        self.score_lbl.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 8))

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        side = tk.Frame(body, bg=PANEL, width=250, padx=12, pady=12)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        side.pack_propagate(False)
        self._build_sidebar(side)

        centre = tk.Frame(body, bg=BG)
        centre.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_viewer(centre)

        right = tk.Frame(body, bg=PANEL, width=310, padx=10, pady=10)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        right.pack_propagate(False)
        self._build_diagnostics(right)

    def _section(self, parent, text):
        tk.Label(parent, text=text, bg=parent.cget("bg"), fg=ACCENT,
                 font=("Courier", 7, "bold"), anchor="w").pack(fill=tk.X, pady=(10, 4))

    def _build_sidebar(self, p):
        self._section(p, "INPUT")
        tk.Button(p, text="Load MRI slice", command=self._load, bg="#1e293b", fg=FG,
                  relief=tk.FLAT, font=MONO, cursor="hand2", pady=6).pack(fill=tk.X)
        self.file_lbl = tk.Label(p, text="No image loaded.", bg=PANEL, fg=DIM,
                                 font=("Courier", 7), anchor="w", wraplength=225, justify=tk.LEFT)
        self.file_lbl.pack(fill=tk.X, pady=(4, 0))

        self._section(p, "METHOD")
        for val, lab in (("esrg", "ESRG (proposed)"), ("srg", "SRG (Adams & Bischof)")):
            tk.Radiobutton(p, text=lab, variable=self.method, value=val, bg=PANEL, fg=FG,
                           selectcolor="#1e293b", activebackground=PANEL, font=("Courier", 8),
                           anchor="w").pack(fill=tk.X)

        self._section(p, "SEEDING")
        for val, lab in (("auto", "Automatic (Objective 1)"), ("manual", "Manual clicks")):
            tk.Radiobutton(p, text=lab, variable=self.seed_mode, value=val, bg=PANEL, fg=FG,
                           selectcolor="#1e293b", activebackground=PANEL, font=("Courier", 8),
                           anchor="w").pack(fill=tk.X)
        tk.Button(p, text="Clear manual seeds", command=self._clear_seeds, bg="#1e293b",
                  fg=DIM, relief=tk.FLAT, font=("Courier", 8), pady=3).pack(fill=tk.X, pady=(3, 0))

        self._section(p, "ABLATION (E4)")
        for var, lab in ((self.use_local, "Local log measure (Obj 2)"),
                         (self.use_stop, "Adaptive stopping (Obj 3)"),
                         (self.use_n4, "N4 bias correction")):
            tk.Checkbutton(p, text=lab, variable=var, bg=PANEL, fg=FG, selectcolor="#1e293b",
                           activebackground=PANEL, font=("Courier", 8), anchor="w").pack(fill=tk.X)

        self._section(p, "PARAMETERS")
        self._slider(p, "k_L (stopping)", self.k_local, 0.5, 4.0, 0.1)
        self._slider(p, "r (window)", self.radius, 1, 8, 1)
        self._slider(p, "K (Otsu classes)", self.classes, 2, 5, 1)

        self._section(p, "DISPLAY")
        self._slider(p, "Overlay opacity", self.opacity, 0.0, 1.0, 0.05, self._redraw)
        tk.Checkbutton(p, text="Show TP/FP/FN colours", variable=self.error_mode, bg=PANEL,
                       fg=FG, selectcolor="#1e293b", activebackground=PANEL,
                       font=("Courier", 8), anchor="w", command=self._redraw).pack(fill=tk.X)

        self.run_btn = tk.Button(p, text="RUN", command=self._run, bg="#0c2a4a", fg=ACCENT,
                                 relief=tk.FLAT, font=("Courier", 10, "bold"),
                                 cursor="hand2", pady=10)
        self.run_btn.pack(fill=tk.X, pady=(14, 0))
        tk.Button(p, text="Save outputs", command=self._save, bg="#1e293b", fg=DIM,
                  relief=tk.FLAT, font=("Courier", 8), pady=4).pack(fill=tk.X, pady=(4, 0))
        self.status_lbl = tk.Label(p, text="", bg=PANEL, fg=ACCENT, font=("Courier", 8),
                                   anchor="w", wraplength=225, justify=tk.LEFT)
        self.status_lbl.pack(fill=tk.X, pady=(6, 0))

    def _slider(self, p, label, var, lo, hi, res, cmd=None):
        tk.Label(p, text=label, bg=PANEL, fg=DIM, font=("Courier", 7), anchor="w").pack(fill=tk.X)
        tk.Scale(p, variable=var, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                 bg=PANEL, fg=DIM, troughcolor="#1e293b", highlightthickness=0,
                 font=("Courier", 7), command=(lambda _: cmd()) if cmd else None).pack(fill=tk.X)

    def _build_viewer(self, p):
        self.stage_bar = tk.Frame(p, bg=BG)
        self.stage_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        self.stage_buttons = {}

        wrap = tk.Frame(p, bg=PANEL, bd=1, relief=tk.SOLID)
        wrap.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(wrap, bg="#0c1526", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

    def _build_diagnostics(self, p):
        tk.Label(p, text="STAGE DIAGNOSTICS", bg=PANEL, fg=ACCENT,
                 font=("Courier", 7, "bold"), anchor="w").pack(fill=tk.X)
        frame = tk.Frame(p, bg=PANEL)
        frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        sb = tk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.diag = tk.Text(frame, bg="#0c1526", fg=FG, font=("Courier", 8), wrap=tk.WORD,
                            relief=tk.FLAT, yscrollcommand=sb.set, padx=8, pady=8)
        self.diag.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.diag.yview)
        self.diag.insert("1.0", "Run the pipeline to see per-stage diagnostics.\n")
        self.diag.config(state=tk.DISABLED)

    # ── Actions ──────────────────────────────────────────────────────────────
    def _load(self):
        path = filedialog.askopenfilename(
            title="Select MRI slice",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if not path:
            return
        self.image_path = path
        self.result = None
        self.manual_points = []
        gt = find_mask_for(path)
        self.file_lbl.config(text=os.path.basename(path) +
                             ("\nGround truth found." if gt else "\nNo ground truth mask."))
        self._clear_stages()
        self.score_lbl.config(text="Image loaded. Press RUN.")
        self._redraw()

    def _clear_seeds(self):
        self.manual_points = []
        self.status_lbl.config(text="Manual seeds cleared.")
        self._redraw()

    def _on_click(self, event):
        """In manual mode, a click plants a seed at that pixel."""
        if self.seed_mode.get() != "manual" or not self.image_path:
            return
        rc = self._canvas_to_image(event.x, event.y)
        if rc:
            self.manual_points.append(rc)
            self.status_lbl.config(text=f"{len(self.manual_points)} manual seed(s).")
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
            messagebox.showwarning("No image", "Load an MRI slice first.")
            return
        if self.seed_mode.get() == "manual" and not self.manual_points:
            messagebox.showwarning("No seeds", "Click on the tumor to plant a seed.")
            return
        cfg = self._current_config()
        self.run_btn.config(state=tk.DISABLED, text="Running…")

        def worker():
            try:
                res = run(self.image_path, cfg, manual_points=self.manual_points,
                          progress=lambda m: self.root.after(0, self.status_lbl.config, {"text": m}))
                self.root.after(0, self._done, res, None)
            except Exception as e:
                self.root.after(0, self._done, None, e)

        threading.Thread(target=worker, daemon=True).start()

    def _done(self, res, err):
        self.run_btn.config(state=tk.NORMAL, text="RUN")
        if err:
            self.status_lbl.config(text="Failed.")
            messagebox.showerror("Pipeline error", str(err))
            return
        self.result = res
        self.status_lbl.config(text=res.status)
        self._build_stage_buttons()
        self.current = "final"
        self._select("final")
        self.score_lbl.config(
            text=viz.score_line(res.scores) if res.status != "NO TUMOR CANDIDATE"
            else "NO TUMOR CANDIDATE — no seed survived interior filtering.",
            fg=FAIL if res.status != "OK" else FG)

    def _clear_stages(self):
        for w in self.stage_bar.winfo_children():
            w.destroy()
        self.stage_buttons = {}

    def _build_stage_buttons(self):
        """One button per stage; colour encodes that stage's status."""
        self._clear_stages()
        for st in self.result.stages:
            colour = {"OK": DIM, "WARN": WARN, "FAIL": FAIL}[st.status]
            b = tk.Button(self.stage_bar, text=st.name, bg="#1e293b", fg=colour,
                          relief=tk.FLAT, font=("Courier", 8), cursor="hand2", padx=8, pady=5,
                          command=lambda k=st.key: self._select(k))
            b.pack(side=tk.LEFT, padx=(0, 4))
            self.stage_buttons[st.key] = b

    def _select(self, key):
        self.current = key
        for k, b in self.stage_buttons.items():
            b.config(bg="#0c2a4a" if k == key else "#1e293b")
        self._show_diagnostics(key)
        self._redraw()

    def _show_diagnostics(self, key):
        st = self.result.stage(key)
        self.diag.config(state=tk.NORMAL)
        self.diag.delete("1.0", tk.END)
        if st:
            self.diag.insert(tk.END, f"{st.name}\nstatus: {st.status}   {st.seconds * 1000:.0f} ms\n\n")
            for k, v in st.info.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    self.diag.insert(tk.END, f"{k}:\n")
                    for item in v:
                        self.diag.insert(tk.END, "  " + ", ".join(f"{a}={b}" for a, b in item.items()) + "\n")
                else:
                    self.diag.insert(tk.END, f"{k}: {v}\n")
            if key == "final" and self.result.scores:
                self.diag.insert(tk.END, "\nscores:\n")
                for k, v in self.result.scores.items():
                    self.diag.insert(tk.END, f"  {k}: {v}\n")
        self.diag.config(state=tk.DISABLED)

    # ── Drawing ──────────────────────────────────────────────────────────────
    def _geometry(self):
        if not self.result:
            return None
        h, w = self.result.stage("input").image.shape
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        s = min(cw / w, ch / h)
        return w, h, s, (cw - w * s) / 2, (ch - h * s) / 2

    def _canvas_to_image(self, x, y):
        g = self._geometry()
        if not g:
            return None
        w, h, s, ox, oy = g
        c, r = int((x - ox) / s), int((y - oy) / s)
        return (r, c) if 0 <= r < h and 0 <= c < w else None

    def _redraw(self):
        self.canvas.delete("all")
        if not self.result:
            cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
            self.canvas.create_text(cw // 2, ch // 2, text="Load an MRI slice and press RUN",
                                    fill="#1e3a5a", font=MONO)
            return
        st = self.result.stage(self.current) or self.result.stages[-1]
        base = self.result.stage("input").image

        if st.key == "final":
            rgb = viz.overlay_result(base, st.image, self.result.gt,
                                     self.opacity.get(), self.error_mode.get())
        else:
            rgb = viz.render_stage(st, base, self.result.gt, self.opacity.get())

        if self.seed_mode.get() == "manual" and self.manual_points:
            for r, c in self.manual_points:
                rgb[max(0, r - 2):r + 3, max(0, c - 2):c + 3] = viz.RED

        w, h, s, ox, oy = self._geometry()
        img = Image.fromarray(rgb.astype(np.uint8)).resize(
            (max(1, int(w * s)), max(1, int(h * s))), Image.NEAREST)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self._photo)

    def _save(self):
        """Write the final mask, the annotated overlay and every stage view."""
        if not self.result:
            messagebox.showwarning("Nothing to save", "Run the pipeline first.")
            return
        folder = filedialog.askdirectory(title="Choose an output folder")
        if not folder:
            return
        from esrg.io_utils import save_png
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        base = self.result.stage("input").image
        save_png(self.result.mask, os.path.join(folder, f"{stem}_mask.png"))
        save_png(viz.overlay_result(base, self.result.mask, self.result.gt,
                                    self.opacity.get(), self.error_mode.get()),
                 os.path.join(folder, f"{stem}_overlay.png"))
        for st in self.result.stages:
            save_png(viz.render_stage(st, base, self.result.gt, self.opacity.get()),
                     os.path.join(folder, f"{stem}_stage_{st.key}.png"))
        self._current_config().save(os.path.join(folder, f"{stem}_config.json"))
        self.status_lbl.config(text=f"Saved to {folder}")


if __name__ == "__main__":
    root = tk.Tk()
    ESRGApp(root)
    root.mainloop()
