"""MuscleMap visualization for matplotlib.

Uses the full set of SVG paths extracted from the MuscleMap Swift package
(github.com/melihcolpan/MuscleMap), parsed once into musclemap_paths.json.
"""

import json
import re
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath


class BodyGender(Enum):
    MALE = "male"
    FEMALE = "female"


# MuscleMap viewboxes (from BodyPathData.swift)
VIEWBOXES = {
    ("male", "front"): (0, 95, 727, 1280),
    ("male", "back"): (718, 95, 727, 1280),
    ("female", "front"): (0, 0, 650, 1450),
    ("female", "back"): (823, 0, 650, 1450),
}

# Slugs whose paths should NOT be tinted by intensity (rendered as base outline only).
COSMETIC_SLUGS = {"hair", "head", "hands", "feet", "knees"}

# Mapping from ExRx muscle names (as produced by planner.py) to MuscleMap slugs.
# Substring match, lowercased. First hit wins, so order matters: more specific first.
MUSCLE_NAME_TO_SLUG: list[tuple[str, str]] = [
    ("pectoralis major, clavicular", "upper-chest"),
    ("pectoralis major, sternal", "lower-chest"),
    ("pectoralis", "chest"),
    ("rectus abdominis", "abs"),
    ("obliques", "obliques"),
    ("serratus anterior", "serratus"),
    ("biceps brachii", "biceps"),
    ("triceps brachii", "triceps"),
    ("brachialis", "biceps"),
    ("brachioradialis", "forearm"),
    ("forearm", "forearm"),
    ("wrist", "forearm"),
    ("deltoid, anterior", "front-deltoid"),
    ("deltoid, posterior", "rear-deltoid"),
    ("deltoid", "deltoids"),
    ("rotator cuff", "rotator-cuff"),
    ("infraspinatus", "rotator-cuff"),
    ("supraspinatus", "rotator-cuff"),
    ("teres minor", "rotator-cuff"),
    ("subscapularis", "rotator-cuff"),
    ("latissimus dorsi", "upper-back"),
    ("teres major", "upper-back"),
    ("trapezius, lower", "lower-trapezius"),
    ("trapezius, upper", "upper-trapezius"),
    ("trapezius, middle", "trapezius"),
    ("trapezius", "trapezius"),
    ("rhomboids", "rhomboids"),
    ("levator scapulae", "trapezius"),
    ("erector spinae", "lower-back"),
    ("quadratus lumborum", "lower-back"),
    ("gluteus maximus", "gluteal"),
    ("gluteus medius", "gluteal"),
    ("gluteus minimus", "gluteal"),
    ("hip external rotators", "gluteal"),
    ("quadriceps", "quadriceps"),
    ("rectus femoris", "quadriceps"),
    ("vastus", "outer-quad"),
    ("iliopsoas", "hip-flexors"),
    ("hip flexors", "hip-flexors"),
    ("hamstring", "hamstring"),
    ("biceps femoris", "hamstring"),
    ("semimembranosus", "hamstring"),
    ("semitendinosus", "hamstring"),
    ("adductor", "adductors"),
    ("gracilis", "adductors"),
    ("pectineus", "adductors"),
    ("gastrocnemius", "calves"),
    ("soleus", "calves"),
    ("tibialis", "tibialis"),
    ("sternocleidomastoid", "neck"),
    ("neck", "neck"),
]


def map_muscle_name_to_slug(name: str) -> str | None:
    name_l = name.lower()
    for needle, slug in MUSCLE_NAME_TO_SLUG:
        if needle in name_l:
            return slug
    return None


# ---------------------------------------------------------------------------
# SVG path parsing
# ---------------------------------------------------------------------------

# SVG path lexer constants. SVG numbers can be unsigned/signed, with optional
# decimal and exponent. Flags (in arc commands) are exactly one character, '0'
# or '1', and may be glued to neighboring numbers without separators.
_NUM_RE = re.compile(r"[+-]?(?:\d+\.\d+|\.\d+|\d+)(?:[eE][+-]?\d+)?")
_CMD_CHARS = set("MmLlHhVvCcSsQqTtAaZz")


class _PathLexer:
    """Positional SVG path lexer.

    Designed to handle the quirks SVG paths come with:
      - numbers can be glued together with sign as separator (e.g. "1.2-3.4")
      - arc flags are single-character ('0' or '1') and can be glued too
      - whitespace and commas are interchangeable separators
    """

    __slots__ = ("s", "i", "n")

    def __init__(self, s: str):
        self.s = s
        self.i = 0
        self.n = len(s)

    def _skip_sep(self):
        while self.i < self.n and (self.s[self.i].isspace() or self.s[self.i] == ","):
            self.i += 1

    def peek_cmd(self) -> str | None:
        self._skip_sep()
        if self.i < self.n and self.s[self.i] in _CMD_CHARS:
            return self.s[self.i]
        return None

    def take_cmd(self) -> str | None:
        c = self.peek_cmd()
        if c is not None:
            self.i += 1
        return c

    def peek_number(self) -> bool:
        self._skip_sep()
        if self.i >= self.n:
            return False
        ch = self.s[self.i]
        return ch.isdigit() or ch == "." or ch == "-" or ch == "+"

    def take_number(self) -> float | None:
        self._skip_sep()
        if self.i >= self.n:
            return None
        m = _NUM_RE.match(self.s, self.i)
        if not m:
            return None
        self.i = m.end()
        return float(m.group())

    def take_flag(self) -> bool | None:
        """Read a single SVG arc flag character ('0' or '1')."""
        self._skip_sep()
        if self.i >= self.n:
            return None
        ch = self.s[self.i]
        if ch != "0" and ch != "1":
            return None
        self.i += 1
        return ch == "1"


def _arc_to_cubics(
    start: np.ndarray,
    end: np.ndarray,
    rx: float,
    ry: float,
    x_axis_rot_deg: float,
    large_arc: bool,
    sweep: bool,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Convert an SVG elliptical arc to a list of cubic Bezier segments.

    Returns a list of (c1, c2, end) tuples to be appended as CURVE4 vertices.
    Implementation follows the SVG 1.1 implementation notes (Appendix F.6).
    """
    if np.allclose(start, end):
        return []
    if rx == 0 or ry == 0:
        # Degenerate: straight line.
        return [(start, end, end)]

    rx = abs(rx)
    ry = abs(ry)
    phi = np.deg2rad(x_axis_rot_deg)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Step 1: compute (x1', y1')
    dx = (start[0] - end[0]) / 2.0
    dy = (start[1] - end[1]) / 2.0
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy

    # Ensure radii are large enough.
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1:
        s = np.sqrt(lam)
        rx *= s
        ry *= s

    # Step 2: compute (cx', cy')
    sign = -1.0 if large_arc == sweep else 1.0
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    factor = sign * np.sqrt(max(0.0, num / den)) if den != 0 else 0.0
    cxp = factor * (rx * y1p) / ry
    cyp = factor * -(ry * x1p) / rx

    # Step 3: compute (cx, cy)
    cx = cos_phi * cxp - sin_phi * cyp + (start[0] + end[0]) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (start[1] + end[1]) / 2.0

    # Step 4: compute angles theta1 and delta_theta.
    def angle(u, v):
        d = np.dot(u, v)
        len_u = np.linalg.norm(u)
        len_v = np.linalg.norm(v)
        if len_u == 0 or len_v == 0:
            return 0.0
        c = np.clip(d / (len_u * len_v), -1.0, 1.0)
        sgn = 1.0 if (u[0] * v[1] - u[1] * v[0]) >= 0 else -1.0
        return sgn * np.arccos(c)

    v1 = np.array([(x1p - cxp) / rx, (y1p - cyp) / ry])
    v2 = np.array([(-x1p - cxp) / rx, (-y1p - cyp) / ry])
    theta1 = angle(np.array([1.0, 0.0]), v1)
    delta = angle(v1, v2)
    if not sweep and delta > 0:
        delta -= 2 * np.pi
    elif sweep and delta < 0:
        delta += 2 * np.pi

    # Step 5: split into segments of <= 90deg and approximate each with a cubic.
    n = max(1, int(np.ceil(abs(delta) / (np.pi / 2))))
    seg_delta = delta / n
    t = (4.0 / 3.0) * np.tan(seg_delta / 4.0)

    segments = []
    cur_start = start.copy()
    for i in range(n):
        a1 = theta1 + i * seg_delta
        a2 = theta1 + (i + 1) * seg_delta

        cos_a1 = np.cos(a1)
        sin_a1 = np.sin(a1)
        cos_a2 = np.cos(a2)
        sin_a2 = np.sin(a2)

        e_x = cx + rx * (cos_phi * cos_a2) - ry * (sin_phi * sin_a2)
        e_y = cy + rx * (sin_phi * cos_a2) + ry * (cos_phi * sin_a2)
        e = np.array([e_x, e_y])

        # Tangent at start of seg.
        d1_x = -rx * cos_phi * sin_a1 - ry * sin_phi * cos_a1
        d1_y = -rx * sin_phi * sin_a1 + ry * cos_phi * cos_a1
        c1 = cur_start + t * np.array([d1_x, d1_y])

        # Tangent at end of seg (negated).
        d2_x = -rx * cos_phi * sin_a2 - ry * sin_phi * cos_a2
        d2_y = -rx * sin_phi * sin_a2 + ry * cos_phi * cos_a2
        c2 = e - t * np.array([d2_x, d2_y])

        segments.append((c1, c2, e))
        cur_start = e

    return segments


def svg_to_mpl_path(path_str: str) -> MplPath | None:
    """Convert an SVG path string into a matplotlib Path.

    Implements: M/m, L/l, H/h, V/v, C/c, S/s, Q/q, T/t, A/a, Z. Arcs are
    converted to cubic Bezier segments. Implicit moveto-as-lineto and
    continuation of last command are honored.
    """
    verts: list[tuple[float, float]] = []
    codes: list[int] = []

    cur = np.array([0.0, 0.0])
    start = np.array([0.0, 0.0])
    last_ctrl: np.ndarray | None = None  # for S/T smoothing
    last_cmd: str | None = None

    lex = _PathLexer(path_str)
    pending_cmd: str | None = None

    def take_point(rel: bool) -> np.ndarray | None:
        x = lex.take_number()
        y = lex.take_number()
        if x is None or y is None:
            return None
        p = np.array([x, y])
        return cur + p if rel else p

    while True:
        cmd = lex.take_cmd()
        if cmd is not None:
            pending_cmd = cmd
        elif pending_cmd is None:
            # No command and no pending => end (or just whitespace left).
            if not lex.peek_number():
                break
            # Stray numbers without a leading command — skip them.
            lex.take_number()
            continue

        # If we did not just take a command and there are no numbers, end.
        if cmd is None and not lex.peek_number():
            break

        rel = pending_cmd.islower()
        u = pending_cmd.upper()

        if u == "M":
            p = take_point(rel)
            if p is None:
                break
            cur = p
            start = p.copy()
            verts.append(tuple(cur))
            codes.append(MplPath.MOVETO)
            # Subsequent implicit pairs become lineto (per SVG spec).
            while lex.peek_number():
                p = take_point(rel)
                if p is None:
                    break
                cur = p
                verts.append(tuple(cur))
                codes.append(MplPath.LINETO)
            last_ctrl = None
            last_cmd = u
            # Continue command is L (or l if relative was used).
            pending_cmd = "l" if rel else "L"

        elif u == "L":
            while lex.peek_number():
                p = take_point(rel)
                if p is None:
                    break
                cur = p
                verts.append(tuple(cur))
                codes.append(MplPath.LINETO)
            last_ctrl = None
            last_cmd = u

        elif u == "H":
            while lex.peek_number():
                x = lex.take_number()
                if x is None:
                    break
                cur = np.array([cur[0] + x if rel else x, cur[1]])
                verts.append(tuple(cur))
                codes.append(MplPath.LINETO)
            last_ctrl = None
            last_cmd = u

        elif u == "V":
            while lex.peek_number():
                y = lex.take_number()
                if y is None:
                    break
                cur = np.array([cur[0], cur[1] + y if rel else y])
                verts.append(tuple(cur))
                codes.append(MplPath.LINETO)
            last_ctrl = None
            last_cmd = u

        elif u == "C":
            while lex.peek_number():
                c1 = take_point(rel)
                c2 = take_point(rel)
                end = take_point(rel)
                if c1 is None or c2 is None or end is None:
                    break
                verts.extend([tuple(c1), tuple(c2), tuple(end)])
                codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
                cur = end
                last_ctrl = c2
            last_cmd = u

        elif u == "S":
            while lex.peek_number():
                c1 = (
                    2 * cur - last_ctrl
                    if last_ctrl is not None and last_cmd in ("C", "S")
                    else cur
                )
                c2 = take_point(rel)
                end = take_point(rel)
                if c2 is None or end is None:
                    break
                verts.extend([tuple(c1), tuple(c2), tuple(end)])
                codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
                cur = end
                last_ctrl = c2
            last_cmd = u

        elif u == "Q":
            while lex.peek_number():
                c1 = take_point(rel)
                end = take_point(rel)
                if c1 is None or end is None:
                    break
                verts.extend([tuple(c1), tuple(end)])
                codes.extend([MplPath.CURVE3, MplPath.CURVE3])
                cur = end
                last_ctrl = c1
            last_cmd = u

        elif u == "T":
            while lex.peek_number():
                c1 = (
                    2 * cur - last_ctrl
                    if last_ctrl is not None and last_cmd in ("Q", "T")
                    else cur
                )
                end = take_point(rel)
                if end is None:
                    break
                verts.extend([tuple(c1), tuple(end)])
                codes.extend([MplPath.CURVE3, MplPath.CURVE3])
                cur = end
                last_ctrl = c1
            last_cmd = u

        elif u == "A":
            while lex.peek_number():
                rx = lex.take_number()
                ry = lex.take_number()
                x_axis_rot = lex.take_number()
                large_arc = lex.take_flag()
                sweep = lex.take_flag()
                end = take_point(rel)
                if rx is None or ry is None or x_axis_rot is None:
                    break
                if large_arc is None or sweep is None or end is None:
                    break
                segs = _arc_to_cubics(cur, end, rx, ry, x_axis_rot, large_arc, sweep)
                if not segs:
                    verts.append(tuple(end))
                    codes.append(MplPath.LINETO)
                else:
                    for c1, c2, e in segs:
                        verts.extend([tuple(c1), tuple(c2), tuple(e)])
                        codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
                cur = end
            last_ctrl = None
            last_cmd = u

        elif u == "Z":
            verts.append(tuple(start))
            codes.append(MplPath.CLOSEPOLY)
            cur = start.copy()
            last_ctrl = None
            last_cmd = u

        else:
            # Unknown command: drop the pending and continue.
            pending_cmd = None

    if not verts:
        return None
    return MplPath(np.array(verts), np.array(codes))


# ---------------------------------------------------------------------------
# MuscleMap renderer
# ---------------------------------------------------------------------------

_PATH_DB_CACHE: dict | None = None


def _load_path_db() -> dict:
    global _PATH_DB_CACHE
    if _PATH_DB_CACHE is None:
        json_path = Path(__file__).parent / "musclemap_paths.json"
        _PATH_DB_CACHE = json.loads(json_path.read_text())
    return _PATH_DB_CACHE


def _workout_cmap() -> LinearSegmentedColormap:
    """Reproduces MuscleMap's 'workout' color scale: gray -> yellow -> orange -> red."""
    return LinearSegmentedColormap.from_list(
        "mm_workout",
        ["#cfcfcf", "#f5d000", "#f59e0b", "#dc2626"],
    )


class MuscleMapReal:
    """Render a MuscleMap-style heatmap using matplotlib."""

    BASE_FILL = "#e7e2dc"
    BASE_EDGE = "#5b5b5b"
    BASE_LINEWIDTH = 0.6

    def __init__(self, gender: BodyGender = BodyGender.MALE, figsize=(12, 10)):
        self.gender = gender
        self.figsize = figsize
        self.cmap = _workout_cmap()
        self.db = _load_path_db()
        self.fig = None
        self.axes: dict[str, plt.Axes] = {}
        self._setup_figure()
        self._draw_base_silhouette()

    def _setup_figure(self):
        self.fig, (ax_front, ax_back) = plt.subplots(1, 2, figsize=self.figsize)
        self.axes = {"front": ax_front, "back": ax_back}
        for side, ax in self.axes.items():
            x, y, w, h = VIEWBOXES[(self.gender.value, side)]
            ax.set_xlim(x, x + w)
            ax.set_ylim(y + h, y)  # invert y for SVG coordinate system
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title("FRONT" if side == "front" else "BACK", fontsize=13, weight="bold")

    def _draw_base_silhouette(self):
        """Draw all muscles in a neutral base color so the body shape is visible."""
        gender = self.gender.value
        for slug, gender_map in self.db.items():
            if gender not in gender_map:
                continue
            for side, paths_dict in gender_map[gender].items():
                if side not in self.axes:
                    continue
                ax = self.axes[side]
                fill = "none" if slug == "hair" else self.BASE_FILL
                for kind in ("common", "left", "right"):
                    for raw in paths_dict.get(kind, []):
                        mpl_path = svg_to_mpl_path(raw)
                        if mpl_path is None:
                            continue
                        patch = PathPatch(
                            mpl_path,
                            facecolor=fill,
                            edgecolor=self.BASE_EDGE,
                            linewidth=self.BASE_LINEWIDTH,
                            alpha=0.95,
                        )
                        ax.add_patch(patch)

    def apply_muscle_intensity(self, intensity_data: list[dict]):
        """Color muscles based on workout intensity, replicating MuscleMap heatmap."""
        if not intensity_data:
            return

        # Aggregate per slug (ExRx data has many sub-muscles per MuscleMap group).
        slug_intensity: dict[str, float] = {}
        for entry in intensity_data:
            name = entry.get("muscle", "")
            inten = float(entry.get("total_intensity", 0) or 0)
            if inten <= 0:
                continue
            slug = map_muscle_name_to_slug(name)
            if slug is None or slug in COSMETIC_SLUGS:
                continue
            slug_intensity[slug] = slug_intensity.get(slug, 0.0) + inten

        if not slug_intensity:
            return

        max_int = max(slug_intensity.values())
        gender = self.gender.value

        for slug, total in slug_intensity.items():
            if slug not in self.db:
                continue
            norm = total / max_int if max_int > 0 else 0.0
            color = self.cmap(norm)

            gender_map = self.db[slug]
            if gender not in gender_map:
                # Fall back to the other gender if MuscleMap defines it only there.
                fallback = "female" if gender == "male" else "male"
                if fallback in gender_map:
                    gender_map_for_slug = gender_map[fallback]
                else:
                    continue
            else:
                gender_map_for_slug = gender_map[gender]

            for side, paths_dict in gender_map_for_slug.items():
                if side not in self.axes:
                    continue
                ax = self.axes[side]
                for kind in ("common", "left", "right"):
                    for raw in paths_dict.get(kind, []):
                        mpl_path = svg_to_mpl_path(raw)
                        if mpl_path is None:
                            continue
                        patch = PathPatch(
                            mpl_path,
                            facecolor=color,
                            edgecolor="#5b1a1a",
                            linewidth=0.7,
                            alpha=0.95,
                        )
                        ax.add_patch(patch)
