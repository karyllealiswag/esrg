"""
prepare_brats2015.py — Export a 2D slice subset of BraTS (2015, 2021, 2023 GLI) for the ESRG pipeline.

Purpose : BraTS ships 3D volumes (.mha in 2015, .nii.gz later; 240 x 240 x 155 per
          case), while ESRG and its GUI read 2D images only. This script turns a
          random subset of cases into axial T1c PNG slices plus ground-truth masks,
          in the same images/ + masks/ layout as BRISC so find_mask_for() pairs them.
          --scheme picks the label convention (see SCHEMES); 2015 is the default.
Function : scan() lists an archive (.tar.gz, .zip) or folder and pairs each patient's
          T1c volume with its ground-truth (OT) volume; choose() draws patients with
          a fixed seed (HGG/LGG split); export_patient() normalises the T1c volume
          once, keeps up to N evenly spaced axial slices whose tumor-core area passes
          a minimum, and writes the image, the core mask and an enhancing-only mask.
Notes   : Only the T1c and OT members are read, one patient at a time through a temp
          file, so the archive is never fully extracted and nothing is written to a
          path taken from the archive. BraTS 2015 labels are 1 necrosis, 2 edema, 3
          non-enhancing tumor, 4 enhancing tumor; 2021 has 1, 2, 4; 2023 GLI has 1
          necrotic core, 2 edema, 3 enhancing (checked on the real segmentations).
          masks/ holds the tumor core and masks_enhancing/ the enhancing tumor,
          because edema is not visible on T1c. Areas in manifest.csv are in the
          native voxel grid (before the resize to --size). Standalone (numpy,
          SimpleITK, Pillow); the data is untrusted, so run it with -P (no working
          or script folder on the import path):
               python -P experiments/prepare_brats2015.py --archive <path> --list
               python -P experiments/prepare_brats2015.py --archive <path>
"""
import argparse
import csv
import os
import random
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile

import numpy as np
import SimpleITK as sitk
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 2026                    # same seed as experiments/sampling.py
VOLUME_EXT = (".mha", ".nii.gz", ".nii")
NESTED_EXT = (".zip", ".tar", ".tar.gz", ".tgz")
T1C_RE = re.compile(r"(?i)t1c(?:e|gd)?(?=[._\-]|$)")
OT_RE = re.compile(r"(?i)(?:^|[._\-])(?:OT|seg|gt)(?=[._\-]|$)")
SCHEMES = {                    # scheme -> (tumor-core labels, enhancing label)
    "2015": ((1, 3, 4), 4),    # necrosis + non-enhancing + enhancing tumor
    "2021": ((1, 4), 4),
    "2023": ((1, 3), 3),       # BraTS 2023 GLI: enhancing tumor is label 3
}


# ─── Archive access ──────────────────────────────────────────────────────────
class Source:
    """A .tar/.tar.gz/.zip archive or a folder: list member names, stream chosen ones to a temp dir."""

    def __init__(self, path):
        self.path = path
        if os.path.isdir(path):
            self.kind = "dir"
        elif zipfile.is_zipfile(path):
            self.kind = "zip"
        elif tarfile.is_tarfile(path):
            self.kind = "tar"
        else:
            raise SystemExit(f"{path}: not a folder, .zip or .tar/.tar.gz archive")

    def names(self):
        if self.kind == "dir":
            return [os.path.relpath(os.path.join(d, f), self.path).replace("\\", "/")
                    for d, _, fs in os.walk(self.path) for f in fs]
        if self.kind == "zip":
            with zipfile.ZipFile(self.path) as z:
                return [i.filename for i in z.infolist() if not i.is_dir()]
        with tarfile.open(self.path, "r|*") as t:      # stream mode: one sequential pass
            return [m.name for m in t if m.isfile()]

    def fetch(self, wanted, dest):
        """Yield (name, local_path, is_temp) for each wanted member, in archive order."""
        wanted, n = set(wanted), 0
        if self.kind == "dir":
            for name in sorted(wanted):
                yield name, os.path.join(self.path, name), False
            return
        if self.kind == "zip":
            with zipfile.ZipFile(self.path) as z:
                for name in sorted(wanted):
                    local = os.path.join(dest, f"{n:05d}_{os.path.basename(name)}")
                    n += 1
                    with z.open(name) as src, open(local, "wb") as out:
                        shutil.copyfileobj(src, out)
                    yield name, local, True
            return
        with tarfile.open(self.path, "r|*") as t:
            for m in t:
                if not (m.isfile() and m.name in wanted):
                    continue
                local = os.path.join(dest, f"{n:05d}_{os.path.basename(m.name)}")
                n += 1
                with t.extractfile(m) as src, open(local, "wb") as out:
                    shutil.copyfileobj(src, out)
                yield m.name, local, True
                if n == len(wanted):
                    return


# ─── Patient discovery and selection ─────────────────────────────────────────
def locate(name):
    """(grade, patient) from a member path such as .../HGG/<patient>/<modality>/<file>.mha."""
    parts = name.replace("\\", "/").split("/")
    base = parts[-1]
    g = next((i for i, p in enumerate(parts[:-1]) if p.upper() in ("HGG", "LGG")), None)
    if g is not None and g + 1 < len(parts) - 1:
        return parts[g].upper(), parts[g + 1]
    stem = re.sub(r"(?i)\.(mha|nii\.gz|nii)$", "", base)
    folder = parts[-2] if len(parts) > 1 else "root"
    if folder == stem and len(parts) > 2:              # BraTS 2015: folder named like the file
        folder = parts[-3]
    return "UNK", folder


def scan(names):
    """{(grade, patient): {'t1c': name, 'ot': name}} for patients that have both volumes."""
    found = {}
    for n in names:
        if not n.lower().endswith(VOLUME_EXT):
            continue
        base = os.path.basename(n)
        kind = "ot" if OT_RE.search(base) else "t1c" if T1C_RE.search(base) else None
        if kind:
            found.setdefault(locate(n), {}).setdefault(kind, n)
    return {k: v for k, v in found.items() if "t1c" in v and "ot" in v}


def subject_of(patient):
    """Subject ID of a case folder (BraTS-XXX-00307-000 -> 00307; BraTS 2015 TCIA pat153_0002 -> 153)."""
    m = re.search(r"-(\d{5})-\d{3}$", patient) or re.search(r"pat(\d+)_", patient)
    return m.group(1) if m else patient


def choose(patients, n, lgg_fraction, seed, one_per_subject=False):
    """Seeded draw of n patients, about lgg_fraction of them LGG (all grades pooled if none are labelled).
    one_per_subject keeps a single random scan of each subject, so repeat scans are never both drawn."""
    rng = random.Random(seed)
    if one_per_subject:
        by_subject = {}
        for key in sorted(patients):
            by_subject.setdefault((key[0], subject_of(key[1])), []).append(key)
        picked = {}
        for scans in by_subject.values():
            k = rng.choice(scans)
            picked[k] = patients[k]
        patients = picked
    by_grade = {}
    for key in sorted(patients):
        by_grade.setdefault(key[0], []).append(key)
    if "HGG" not in by_grade and "LGG" not in by_grade:
        pool = sorted(patients)
        return rng.sample(pool, min(n, len(pool)))
    lgg, hgg = by_grade.get("LGG", []), by_grade.get("HGG", [])
    n_lgg = min(len(lgg), round(n * lgg_fraction))
    n_hgg = min(len(hgg), n - n_lgg)
    return rng.sample(hgg, n_hgg) + rng.sample(lgg, n_lgg)


# ─── Export ──────────────────────────────────────────────────────────────────
def _resize(a, size, resample):
    img = Image.fromarray(a)
    scale = size / max(img.size)
    return img.resize((round(img.width * scale), round(img.height * scale)), resample)


def export_patient(key, t1c_path, ot_path, out, per_patient, min_area, size, flip_v, flip_h,
                   core_labels, enh_label, prefix):
    """Write up to per_patient axial slices of one patient; return their manifest rows."""
    grade, patient = key
    vol = sitk.GetArrayFromImage(sitk.ReadImage(t1c_path)).astype(np.float32)   # (z, y, x)
    lab = sitk.GetArrayFromImage(sitk.ReadImage(ot_path)).astype(np.int16)
    if vol.ndim != 3 or vol.shape != lab.shape:
        print(f"  skip {patient}: T1c {vol.shape} and labels {lab.shape} do not match")
        return []
    core, enh = np.isin(lab, core_labels), lab == enh_label
    eligible = np.flatnonzero(core.sum(axis=(1, 2)) >= min_area)
    if eligible.size == 0:
        print(f"  skip {patient}: no slice with a tumor core of at least {min_area} px")
        return []
    k = min(per_patient, eligible.size)
    picks = eligible[np.unique(np.linspace(0, eligible.size - 1, k).round().astype(int))]
    nz = vol[vol > 0]                                   # skull-stripped: background is 0
    lo, hi = np.percentile(nz, [0.5, 99.5]) if nz.size else (0.0, 1.0)
    safe = re.sub(r"[^A-Za-z0-9_]+", "-", patient)
    rows = []
    for z in picks:
        img8 = (np.clip((vol[z] - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        planes = [img8, core[z].astype(np.uint8) * 255, enh[z].astype(np.uint8) * 255]
        if flip_v:
            planes = [p[::-1] for p in planes]
        if flip_h:
            planes = [p[:, ::-1] for p in planes]
        tag = "" if grade == "UNK" else grade + "_"
        stem = f"{prefix}_{tag}{safe}_ax_{z:03d}_t1c"
        _resize(np.ascontiguousarray(planes[0]), size, Image.BILINEAR).save(
            os.path.join(out, "images", stem + ".png"))
        _resize(np.ascontiguousarray(planes[1]), size, Image.NEAREST).save(
            os.path.join(out, "masks", stem + ".png"))
        _resize(np.ascontiguousarray(planes[2]), size, Image.NEAREST).save(
            os.path.join(out, "masks_enhancing", stem + ".png"))
        rows.append({"file": stem + ".png", "grade": grade, "patient": patient, "slice": int(z),
                     "core_area": int(core[z].sum()), "enhancing_area": int(enh[z].sum())})
    return rows


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Export a 2D T1c slice subset of BraTS (2015, 2021, 2023 GLI).")
    ap.add_argument("--archive", required=True, help=".tar.gz / .zip archive, or a folder of extracted files")
    ap.add_argument("--scheme", choices=sorted(SCHEMES), default="2015", help="label convention of the data")
    ap.add_argument("--prefix", default=None, help="file-name prefix (default: brats<scheme>)")
    ap.add_argument("--out", default=None, help="output folder (default: <repo>/brats<scheme>)")
    ap.add_argument("--patients", type=int, default=100)
    ap.add_argument("--per-patient", type=int, default=20, help="max axial slices per patient")
    ap.add_argument("--lgg-fraction", type=float, default=0.2)
    ap.add_argument("--one-per-subject", action="store_true", help="draw at most one scan per subject")
    ap.add_argument("--min-area", type=int, default=150, help="min tumor-core pixels (native grid) for a slice to qualify")
    ap.add_argument("--size", type=int, default=512, help="long side of the exported PNGs")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--flip-v", action="store_true", help="flip slices top-bottom")
    ap.add_argument("--flip-h", action="store_true", help="flip slices left-right")
    ap.add_argument("--tmp", default=None, help="folder for temporary volumes (default: system temp)")
    ap.add_argument("--list", action="store_true", help="only report what the archive contains")
    ap.add_argument("--force", action="store_true", help="overwrite an existing export")
    args = ap.parse_args()
    prefix = args.prefix or f"brats{args.scheme}"
    args.out = args.out or os.path.join(ROOT, prefix)
    core_labels, enh_label = SCHEMES[args.scheme]

    src = Source(args.archive)
    names = src.names()
    patients = scan(names)
    grades = {}
    for g, _ in patients:
        grades[g] = grades.get(g, 0) + 1
    vols = [n for n in names if n.lower().endswith(VOLUME_EXT)]
    print(f"{len(names)} files, {len(vols)} volumes, {len(patients)} patients with T1c + OT {grades}")
    if not patients:
        nested = [n for n in names if n.lower().endswith(NESTED_EXT)]
        if nested:
            print("The archive holds nested archives; extract them into an empty folder first and pass that "
                  "folder as --archive:")
            print("  " + "\n  ".join(nested[:10]))
        else:
            print("No T1c + OT pairs recognised. First members:")
            print("  " + "\n  ".join(names[:15]))
        return 1
    if args.list:
        for key in sorted(patients)[:3]:
            print(f"  {key}: {patients[key]['t1c']}  |  {patients[key]['ot']}")
        return 0

    manifest = os.path.join(args.out, "manifest.csv")
    if os.path.exists(manifest) and not args.force:
        raise SystemExit(f"{manifest} exists; pass --force to overwrite the export")
    for sub in ("images", "masks", "masks_enhancing"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    chosen = choose(patients, args.patients, args.lgg_fraction, args.seed, args.one_per_subject)
    wanted = {}
    for key in chosen:
        wanted[patients[key]["t1c"]] = (key, "t1c")
        wanted[patients[key]["ot"]] = (key, "ot")
    print(f"exporting {len(chosen)} patients (up to {args.per_patient} slices each) to {args.out}")

    tmp = tempfile.mkdtemp(prefix="brats2015_", dir=args.tmp)
    rows, pending = [], {}
    try:
        for name, local, is_temp in src.fetch(wanted, tmp):
            key, kind = wanted[name]
            pending.setdefault(key, {})[kind] = (local, is_temp)
            if len(pending[key]) < 2:
                continue
            pair = pending.pop(key)
            try:
                got = export_patient(key, pair["t1c"][0], pair["ot"][0], args.out, args.per_patient,
                                     args.min_area, args.size, args.flip_v, args.flip_h,
                                     core_labels, enh_label, prefix)
                rows += got
                print(f"  {key[0]} {key[1]}: {len(got)} slices ({len(rows)} total)")
            finally:
                for path, temp in pair.values():
                    if temp and os.path.exists(path):
                        os.remove(path)
        if pending:
            print(f"warning: {len(pending)} patients missing a T1c or OT member were skipped")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    rows.sort(key=lambda r: r["file"])
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "grade", "patient", "slice", "core_area", "enhancing_area"])
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(rows)} slices from {len({(r['grade'], r['patient']) for r in rows})} patients -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
