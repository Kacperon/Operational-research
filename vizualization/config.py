"""Muscle mapping, scoring weights, and color configuration."""
from __future__ import annotations

# ─── 3 unified roles ────────────────────────────────────────────
# CSV columns are merged into three display categories:
#   Primary    ← Target
#   Secondary  ← Synergists
#   Stabilizer ← Dynamic Stabilizers + Stabilizers + Antagonist Stabilizers

ROLE_COLUMNS = [
	"Target",
	"Synergists",
	"Dynamic Stabilizers",
	"Stabilizers",
	"Antagonist Stabilizers",
]

# Maps each CSV column to a display role
CSV_TO_ROLE: dict[str, str] = {
	"Target":                  "Primary",
	"Synergists":              "Secondary",
	"Dynamic Stabilizers":     "Stabilizer",
	"Stabilizers":             "Stabilizer",
	"Antagonist Stabilizers":  "Stabilizer",
}

DISPLAY_ROLES = ["Primary", "Secondary", "Stabilizer"]

ROLE_WEIGHTS: dict[str, float] = {
	"Primary":    1.0,
	"Secondary":  0.6,
	"Stabilizer": 0.25,
}

ROLE_PRIORITY: dict[str, int] = {
	"Primary":    0,
	"Secondary":  1,
	"Stabilizer": 2,
}

# ─── Role-based colors (single-exercise view) ────────────────────

ROLE_COLORS: dict[str, tuple[str, float]] = {
	#                       (facecolor,   alpha)
	"Primary":    ("#b60000",   0.85),   # deep red
	"Secondary":  ("#ff8800",   0.70),   # orange
	"Stabilizer": ("#ffe600",   0.45),   # yellow
}

ROLE_EDGE_COLOR: str = "#4a1010"

# Inactive muscle patch style
INACTIVE_FACE = "#e8e8e8"
INACTIVE_EDGE = "#999999"
INACTIVE_ALPHA = 0.15

# ─── SVG group splitting ─────────────────────────────────────────
# The simplified SVG lumps several anatomical muscles into one <g>.
# GROUP_SPLITS tells the loader which path indices belong to which
# sub-region.  Any group listed here is replaced by its sub-regions;
# the original group ID is no longer used.
#
# Path indices were determined from the SVG geometry:
#   calves  (8 paths): 0,1 front upper (tibialis anterior)
#                       6,7 front lower (soleus, visible from front)
#                       2,3,4,5 rear (gastrocnemius)
#   hamstrings (4 paths): 0,1 narrow/medial (semitendinosus+semimembranosus)
#                          2,3 wide/lateral (biceps femoris)
#   quads (10 paths): 0,5 front large (rectus femoris)
#                      1,4 front outer (vastus lateralis)
#                      2,3 front inner-low (vastus medialis)
#                      6,7,8,9 rear visibility

GROUP_SPLITS: dict[str, dict[str, list[int]]] = {
	"calves": {
		"calves_gastrocnemius": [2, 3, 4, 5],
		"calves_soleus":        [6, 7],
		"calves_tibialis":      [0, 1],
	},
	"hamstrings": {
		"hamstrings_medial":  [0, 1],   # semitendinosus / semimembranosus
		"hamstrings_lateral": [2, 3],   # biceps femoris
	},
	"quads": {
		"quads_rectus":  [0, 5],              # rectus femoris (central)
		"quads_vastus_l": [1, 4],             # vastus lateralis (outer)
		"quads_vastus_m": [2, 3],             # vastus medialis (inner-low)
		"quads_rear":     [6, 7, 8, 9],       # rear visibility
	},
}

# ─── SVG group IDs ────────────────────────────────────────────────
# These are the final region IDs used for coloring — includes both
# un-split groups and sub-regions produced by GROUP_SPLITS.

MUSCLE_SVG_IDS: set[str] = {
	# Unsplit groups
	"obliques", "lower_abs", "upper_abs", "biceps",
	"side_delts", "front_delts", "upper_pecs", "rear_delts",
	"lower_pecs", "middle_pecs", "rhomboids",
	"lower_back", "hip_abductor", "neck", "upper_traps",
	"lower_traps", "forearms", "triceps", "glutes",
	"lats", "hip_adductor",
	# Split: calves
	"calves_gastrocnemius", "calves_soleus", "calves_tibialis",
	# Split: hamstrings
	"hamstrings_medial", "hamstrings_lateral",
	# Split: quads
	"quads_rectus", "quads_vastus_l", "quads_vastus_m", "quads_rear",
}

STRUCTURE_SVG_IDS: set[str] = {
	"front_borders", "rear_borders", "front", "rear", "face",
}

# ─── Muscle name → SVG group mapping ─────────────────────────────
# Keys are lowercase, whitespace-collapsed muscle names as they
# appear in the CSV (after normalization).  Values are SVG region
# ids (may be sub-regions from GROUP_SPLITS).

MUSCLE_TO_SVG: dict[str, list[str]] = {
	# ── Neck ──────────────────────────────────────────────────────
	"trapezius, upper":       ["upper_traps", "neck"],
	"trapezius, middle":      ["lower_traps", "rhomboids"],
	"trapezius, lower":       ["lower_traps"],
	"levator scapulae":       ["neck"],
	"sternocleidomastoid":    ["neck"],
	"longus capitis":         ["neck"],
	"longus colli":           ["neck"],
	"rectus capitus":         ["neck"],
	"splenius":               ["neck"],
	# ── Shoulders ─────────────────────────────────────────────────
	"deltoid, anterior":      ["front_delts"],
	"deltoid, lateral":       ["side_delts"],
	"deltoid, posterior":     ["rear_delts"],
	"supraspinatus":          ["side_delts"],
	"infraspinatus":          ["rear_delts"],
	"subscapularis":          ["front_delts"],
	"teres minor":            ["rear_delts"],
	"teres major":            ["rear_delts", "lats"],
	# ── Chest ─────────────────────────────────────────────────────
	"pectoralis major":              ["upper_pecs", "middle_pecs", "lower_pecs"],
	"pectoralis major, clavicular":  ["upper_pecs"],
	"pectoralis major, sternal":     ["middle_pecs", "lower_pecs"],
	"pectoralis minor":              ["upper_pecs"],
	"serratus anterior":             ["obliques"],
	# ── Arms – biceps ────────────────────────────────────────────
	"biceps brachii":            ["biceps"],
	"biceps brachii, short head": ["biceps"],
	"brachialis":                ["biceps"],
	"coracobrachialis":          ["biceps"],
	# ── Arms – triceps ───────────────────────────────────────────
	"triceps brachii":           ["triceps"],
	"triceps brachii, long head": ["triceps"],
	"triceps, long head":        ["triceps"],
	"triceps":                   ["triceps"],
	# ── Arms – forearms ──────────────────────────────────────────
	"brachioradialis":        ["forearms"],
	"wrist flexors":          ["forearms"],
	"wrist extensors":        ["forearms"],
	"pronators":              ["forearms"],
	"supinator":              ["forearms"],
	"flexor carpi radialis":  ["forearms"],
	"flexor carpi ulnaris":   ["forearms"],
	"extensor carpi radialis": ["forearms"],
	"extensor carpi ulnaris": ["forearms"],
	# ── Core ──────────────────────────────────────────────────────
	"rectus abdominis":       ["upper_abs", "lower_abs"],
	"obliques":               ["obliques"],
	"transverse abdominis":   ["lower_abs"],
	"quadratus lumborum":     ["lower_back"],
	"erector spinae":         ["lower_back"],
	"iliocastalis lumborum":  ["lower_back"],
	"iliocastalis thoracis":  ["lower_back"],
	"iliopsoas":              ["hip_adductor"],
	"psoas major":            ["hip_adductor"],
	# ── Back ──────────────────────────────────────────────────────
	"latissimus dorsi":       ["lats"],
	"rhomboids":              ["rhomboids"],
	"back, general":          ["lats", "lower_back", "rhomboids"],
	"general, back":          ["lats", "lower_back", "rhomboids"],
	# ── Glutes / Hips ────────────────────────────────────────────
	"gluteus maximus":                ["glutes"],
	"gluteus maximus, lower fibers":  ["glutes"],
	"gluteus medius":                 ["hip_abductor"],
	"gluteus medius, posterior fibers": ["hip_abductor"],
	"gluteus minimus":                ["hip_abductor"],
	"gluteus minimus, anterior fibers": ["hip_abductor"],
	"tensor fasciae latae":           ["hip_abductor"],
	"piriformis":                     ["glutes"],
	"obturator externus":             ["glutes"],
	"obturator internus":             ["glutes"],
	"gemellus inferior":              ["glutes"],
	"gemellus superior":              ["glutes"],
	"quadratus femoris":              ["glutes"],
	"hip external rotators":          ["glutes"],
	"hip abductors":                  ["hip_abductor"],
	"hip internal rotators":          ["hip_abductor"],
	"adductors, hip":                 ["hip_adductor"],
	# ── Legs – quads ─────────────────────────────────────────────
	"quadriceps":             ["quads_rectus", "quads_vastus_l", "quads_vastus_m", "quads_rear"],
	"rectus femoris":         ["quads_rectus"],
	"vastus lateralis":       ["quads_vastus_l"],
	"vastus medialis":        ["quads_vastus_m"],
	"vastus intermedius":     ["quads_rectus", "quads_rear"],
	"sartorius":              ["quads_vastus_m"],
	# ── Legs – adductors ─────────────────────────────────────────
	"adductor magnus":        ["hip_adductor"],
	"adductor longus":        ["hip_adductor"],
	"adductor brevis":        ["hip_adductor"],
	"pectineus":              ["hip_adductor"],
	"gracilis":               ["hip_adductor"],
	# ── Legs – hamstrings ────────────────────────────────────────
	"hamstrings":             ["hamstrings_medial", "hamstrings_lateral"],
	"biceps femoris":         ["hamstrings_lateral"],
	"semitendinosus":         ["hamstrings_medial"],
	"semimembranosus":        ["hamstrings_medial"],
	# ── Lower leg ────────────────────────────────────────────────
	"gastrocnemius":          ["calves_gastrocnemius"],
	"soleus":                 ["calves_soleus"],
	"tibialis anterior":      ["calves_tibialis"],
	"popliteus":              ["calves_gastrocnemius"],
}

# ─── Body-part fallback ──────────────────────────────────────────
# When no specific muscle maps, use the broad body_part field.

BODY_PART_TO_SVG: dict[str, list[str]] = {
	"back":       ["lats", "upper_traps", "lower_traps", "lower_back", "rhomboids"],
	"calves":     ["calves_gastrocnemius", "calves_soleus", "calves_tibialis"],
	"chest":      ["upper_pecs", "middle_pecs", "lower_pecs"],
	"forearms":   ["forearms"],
	"hips":       ["glutes", "hip_abductor", "hip_adductor"],
	"neck":       ["neck"],
	"shoulders":  ["front_delts", "side_delts", "rear_delts"],
	"thighs":     ["quads_rectus", "quads_vastus_l", "quads_vastus_m", "quads_rear",
	               "hamstrings_medial", "hamstrings_lateral"],
	"upper arms": ["biceps", "triceps"],
	"waist":      ["upper_abs", "lower_abs", "obliques", "lower_back"],
}
