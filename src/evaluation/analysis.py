from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots as _scienceplots
from ml_mmr_tracking.analysis import add_rectangular_coil, bootstrap_ci, coils, e50_e95

_ = _scienceplots

plt.style.use(["science", "ieee"])

TEXTWIDTH_PT = 516.0
COLUMNWIDTH_PT = 252.0
PT_PER_IN = 72.27

FIGURE_MODE = (
    "two-column"  # "two-column" -> IEEE text width, "one-column" -> IEEE column width
)
FIGWIDTH = (TEXTWIDTH_PT if FIGURE_MODE == "two-column" else COLUMNWIDTH_PT) / PT_PER_IN
FIGHEIGHT = FIGWIDTH * 0.4
LAYOUT_PAD_IN = 0.02
LAYOUT_SPACE = 0.06

plt.rcParams.update(
    {
        # scienceplots ieee style may request Times only; keep Times first but add fallbacks.
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "TeX Gyre Termes",
            "DejaVu Serif",
        ],
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    }
)

plt.rcParams.update(
    {
        "text.usetex": True,
    }
)

TRACKING_RESULTS = Path("data/evaluation/inference-results.csv")
GT_FILE_PATH = Path("data/raw/experiment/gt.csv")
OUTPUT_DIR = Path("data/evaluation")

tracking_results = pd.read_csv(TRACKING_RESULTS)
tracking_results[["x", "y", "z"]] *= 1000  # convert to mm
tracking_results = tracking_results[tracking_results["frame"] > 15]
tracking_results = tracking_results.loc[
    tracking_results["method"] == "deep-conv-transformer"
].copy()
tracking_results.drop(columns=["method"], inplace=True)
tracking_results.rename(columns={"x": "X", "y": "Y", "z": "Z"}, inplace=True)

gt = pd.read_csv(GT_FILE_PATH)
gt_means = gt[["X", "Y", "Z"]].mean()

median_tracking_results = tracking_results.groupby(["measurement"], as_index=False)[
    ["X", "Y", "Z"]
].median()
# Center each orientation by its mean offset.
method_means = median_tracking_results.mean(axis=0, numeric_only=True)

# add Z from the conv transformer
gt["Z"] = gt["Z"] - gt_means["Z"] + method_means["Z"]
gt["X"] = gt["X"] - gt_means["X"] + method_means["X"]
gt["Y"] = gt["Y"] - gt_means["Y"] + method_means["Y"]

merged_df = tracking_results.merge(
    gt[
        [
            "measurement",
            "X",
            "Y",
            "Z",
        ]
    ],
    on="measurement",
    how="left",
    suffixes=("", "_gt"),
)

merged_df["distance_gt"] = (
    (merged_df["X"] - merged_df["X_gt"]) ** 2
    + (merged_df["Y"] - merged_df["Y_gt"]) ** 2
    + (merged_df["Z"] - merged_df["Z_gt"]) ** 2
) ** 0.5


errors_3d = merged_df["distance_gt"].dropna().to_numpy(dtype=float)
e50_3d, e95_3d = e50_e95(errors_3d)
ci_3d_e50 = bootstrap_ci(errors_3d, np.median)
ci_3d_e95 = bootstrap_ci(errors_3d, lambda x: np.quantile(x, 0.95))

print("3D error")
print(f"E50: {e50_3d:.2f} mm, 95% CI: ({ci_3d_e50[0]:.2f}, {ci_3d_e50[1]:.2f})")
print(f"E95: {e95_3d:.2f} mm, 95% CI: ({ci_3d_e95[0]:.2f}, {ci_3d_e95[1]:.2f})")


fig = plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
grid = fig.add_gridspec(
    3,
    4,
    width_ratios=[2, 0.01, 0.6, 0.3],
    height_ratios=[0.3, 0.6, 0.05],
    wspace=0.1 * FIGHEIGHT / FIGWIDTH,
    hspace=0.1,
)

ax_3d = fig.add_subplot(grid[:, 0:2], projection="3d")
ax_xz = fig.add_subplot(grid[0, 2])
ax_yz = fig.add_subplot(grid[1, 3])
ax_xy = fig.add_subplot(grid[1, 2], sharex=ax_xz, sharey=ax_yz)
ax_empty = fig.add_subplot(grid[0, 3])
ax_empty.set_axis_off()

ax_3d.scatter(
    merged_df["X"],
    merged_df["Y"],
    merged_df["Z"],
    s=3,
    alpha=0.6,
    label="Proposed",
    color="#0C5DA5",
)
ax_xy.scatter(merged_df["X"], merged_df["Y"], s=3, alpha=0.6, color="#0C5DA5")
ax_xz.scatter(merged_df["X"], merged_df["Z"], s=3, alpha=0.6, color="#0C5DA5")
ax_yz.scatter(merged_df["Z"], merged_df["Y"], s=3, alpha=0.6, color="#0C5DA5")


ax_3d.scatter(
    merged_df["X_gt"],
    merged_df["Y_gt"],
    merged_df["Z_gt"],
    s=7,
    marker="x",
    color="#333333",
    alpha=0.9,
    label="GT",
)
ax_xy.scatter(
    merged_df["X_gt"],
    merged_df["Y_gt"],
    s=7,
    marker="x",
    color="#333333",
    alpha=0.9,
)
ax_xz.scatter(
    merged_df["X_gt"],
    merged_df["Z_gt"],
    s=7,
    marker="x",
    color="#333333",
    alpha=0.9,
)
ax_yz.scatter(
    merged_df["Z_gt"],
    merged_df["Y_gt"],
    s=7,
    marker="x",
    color="#333333",
    alpha=0.9,
)

ax_xy.set_xlim(-215, 215)
ax_xy.set_ylim(-215, 215)
ax_xy.set_xticks([-200, -100, 0, 100, 200])
ax_xy.set_yticks([-200, -100, 0, 100, 200])
ax_xy.set_xlabel("$x\\text{ / mm}$")
ax_xy.set_ylabel("$y\\text{ / mm}$")
ax_xy.yaxis.set_label_coords(-0.25, 0.5)
ax_xy.xaxis.set_label_coords(0.5, -0.15)

ax_xz.set_yticks([0, 100, 200])
ax_xz.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
ax_xz.set_ylabel("$z\\text{ / mm}$")
# Keep panel-label alignment consistent with the XY panel.
ax_xz.yaxis.set_label_coords(-0.25, 0.5)

ax_yz.set_xticks([0, 100, 200])
ax_yz.tick_params(axis="y", which="both", left=False, labelleft=False)
# ax_yz.label_outer()
ax_yz.set_xlabel("$z\\text{ / mm}$")
ax_yz.xaxis.set_label_coords(0.5, -0.15)

ax_3d.set_xlabel("$x$")
ax_3d.set_ylabel("$y$")
ax_3d.set_zlabel("$z$")

# # Apply 3D plot styling
combined_coords = np.vstack([merged_df[["X_gt", "Y_gt", "Z_gt"]].values])
coord_ranges = np.ptp(combined_coords, axis=0)
coord_ranges = np.where(coord_ranges == 0.0, 1.0, coord_ranges)
ax_3d.set_box_aspect(coord_ranges)

axis_origin = (-280, -280, 0)
axis_length = 100
ax_3d.set_axis_off()
ax_3d.quiver(
    axis_origin[0],
    axis_origin[1],
    axis_origin[2],
    axis_length,
    0.0,
    0.0,
    color="red",
    linewidth=0.8,
    arrow_length_ratio=0.12,
)
ax_3d.quiver(
    axis_origin[0],
    axis_origin[1],
    axis_origin[2],
    0.0,
    axis_length,
    0.0,
    color="green",
    linewidth=0.8,
    arrow_length_ratio=0.12,
)
ax_3d.quiver(
    axis_origin[0],
    axis_origin[1],
    axis_origin[2],
    0.0,
    0.0,
    axis_length,
    color="blue",
    linewidth=0.8,
    arrow_length_ratio=0.12,
)

label_offset = 0.04 * axis_length
ax_3d.text(
    axis_origin[0] + axis_length + label_offset,
    axis_origin[1] - 40,
    axis_origin[2],
    "$x$",
    color="red",
)
ax_3d.text(
    axis_origin[0] - 20,
    axis_origin[1] + axis_length + label_offset - 5,
    axis_origin[2],
    "$y$",
    color="green",
)
ax_3d.text(
    axis_origin[0] - 15,
    axis_origin[1] - 15,
    axis_origin[2] + axis_length + label_offset,
    "$z$",
    color="blue",
)

# Draw coils
for coil in coils:
    add_rectangular_coil(ax_3d, coil)


# Combine legends from all axes into a single horizontal legend below figure
all_handles = []
all_labels = []
for ax in fig.axes:
    h, labels = ax.get_legend_handles_labels()
    for hi, li in zip(h, labels):
        if li and not li.startswith("_"):
            all_handles.append(hi)
            all_labels.append(li)

unique = OrderedDict()
for lab, han in zip(all_labels, all_handles):
    if lab not in unique:
        unique[lab] = han

ncol = len(unique)
fig.legend(
    list(unique.values()),
    list(unique.keys()),
    loc="lower center",
    ncol=ncol,
    frameon=False,
    bbox_to_anchor=(0.5, 0.02),
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path_png = OUTPUT_DIR / "fig2.png"
output_path_pdf = OUTPUT_DIR / "fig2.pdf"

fig.savefig(output_path_png, dpi=300, bbox_inches="tight", pad_inches=0.01)
fig.savefig(output_path_pdf, bbox_inches="tight", pad_inches=0.01)
plt.close(fig)
