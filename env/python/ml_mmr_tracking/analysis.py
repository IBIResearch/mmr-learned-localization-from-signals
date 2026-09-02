import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def e50_e95(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.median(values)), float(np.quantile(values, 0.95))


def bootstrap_ci(values, stat_fn, n_boot=2000, alpha=0.05, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        stat = float(stat_fn(values))
        return stat, stat

    rng = np.random.default_rng(seed)
    stats = np.empty(int(n_boot), dtype=float)
    for idx in range(int(n_boot)):
        sample = rng.choice(values, size=values.size, replace=True)
        stats[idx] = float(stat_fn(sample))

    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def round_to_significant_digits(value, significant_digits=2):
    value = float(value)
    if not np.isfinite(value) or value == 0.0:
        return value, 0

    exponent = int(np.floor(np.log10(abs(value))))
    ndigits = int(significant_digits - exponent - 1)
    return float(round(value, ndigits)), ndigits


# Coil definitions
coils = [
    {
        "center": [-0.061, -0.183, 0.0],
        "sideA": 0.122,
        "sideB": 0.122,
        "thickness": 0.032,
        "windings": 60,
    },
    {
        "center": [0.061, -0.183, 0.0],
        "sideA": 0.122,
        "sideB": 0.122,
        "thickness": 0.032,
        "windings": 60,
    },
    {
        "center": [0.183, -0.183, 0.0],
        "sideA": 0.122,
        "sideB": 0.122,
        "thickness": 0.032,
        "windings": 60,
    },
    {
        "center": [0.061, 0.183, 0.0],
        "sideA": 0.122,
        "sideB": 0.122,
        "thickness": 0.032,
        "windings": 60,
    },
    {
        "center": [-0.061, -0.061, 0.0],
        "sideA": 0.122,
        "sideB": 0.122,
        "thickness": 0.032,
        "windings": 60,
    },
    {
        "center": [-0.183, 0.183, 0.0],
        "sideA": 0.122,
        "sideB": 0.122,
        "thickness": 0.032,
        "windings": 60,
    },
    {
        "center": [-0.061, 0.183, 0.0],
        "sideA": 0.122,
        "sideB": 0.122,
        "thickness": 0.032,
        "windings": 60,
    },
    {
        "center": [-0.183, 0.061, 0.0],
        "sideA": 0.122,
        "sideB": 0.122,
        "thickness": 0.032,
        "windings": 60,
    },
    {
        "center": [0.183, -0.061, 0.0],
        "sideA": 0.122,
        "sideB": 0.122,
        "thickness": 0.032,
        "windings": 60,
    },
    {
        "center": [-0.183, -0.183, 0.0],
        "sideA": 0.122,
        "sideB": 0.122,
        "thickness": 0.032,
        "windings": 60,
    },
    {
        "center": [-0.061, 0.061, 0.0],
        "sideA": 0.122,
        "sideB": 0.122,
        "thickness": 0.032,
        "windings": 60,
    },
    {
        "center": [0.183, 0.183, 0.0],
        "sideA": 0.122,
        "sideB": 0.122,
        "thickness": 0.032,
        "windings": 60,
    },
    {
        "center": [0.061, -0.061, 0.0],
        "sideA": 0.122,
        "sideB": 0.122,
        "thickness": 0.032,
        "windings": 60,
    },
    {
        "center": [-0.183, -0.061, 0.0],
        "sideA": 0.122,
        "sideB": 0.122,
        "thickness": 0.032,
        "windings": 60,
    },
    {
        "center": [0.061, 0.061, 0.0],
        "sideA": 0.122,
        "sideB": 0.122,
        "thickness": 0.032,
        "windings": 60,
    },
    {
        "center": [0.183, 0.061, 0.0],
        "sideA": 0.122,
        "sideB": 0.122,
        "thickness": 0.032,
        "windings": 60,
    },
]


def add_rectangular_coil(ax, coil):
    """Draw a hollow rectangular coil volume as a 3D Poly3DCollection.

    The hollow box is the volume between the outer rectangle and the inner
    rectangle whose distance is `windings * thickness`.
    """
    c = np.array(coil["center"], dtype=float) * 1000
    a = coil["sideA"] * 1000
    b = coil["sideB"] * 1000
    th = coil["thickness"] * 1000

    # Corners in clockwise order
    face1 = [
        [c[0] - a / 2, c[1] - b / 2, c[2]],
        [c[0] + a / 2, c[1] - b / 2, c[2]],
        [c[0] + a / 2 - th, c[1] - b / 2 + th, c[2]],
        [c[0] - a / 2 + th, c[1] - b / 2 + th, c[2]],
        [c[0] - a / 2 + th, c[1] + b / 2 - th, c[2]],
        [c[0] - a / 2, c[1] + b / 2, c[2]],
    ]
    face2 = [
        [c[0] + a / 2, c[1] + b / 2, c[2]],
        [c[0] - a / 2, c[1] + b / 2, c[2]],
        [c[0] - a / 2 + th, c[1] + b / 2 - th, c[2]],
        [c[0] + a / 2 - th, c[1] + b / 2 - th, c[2]],
        [c[0] + a / 2 - th, c[1] - b / 2 + th, c[2]],
        [c[0] + a / 2, c[1] - b / 2, c[2]],
    ]

    # Create flat outer face (blue, translucent)
    col = [0.0, 111 / 255, 41 / 255, 1.0]
    face_color = (col[0], col[1], col[2], min(col[3] if len(col) > 3 else 0.6, 0.6))
    poly_outer = Poly3DCollection(
        [face1, face2],
        facecolors=face_color,
        linewidths=0.2,
        edgecolors=(0, 0, 0, 0.15),
    )
    ax.add_collection3d(poly_outer)
