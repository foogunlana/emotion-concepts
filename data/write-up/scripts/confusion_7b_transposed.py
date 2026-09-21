"""Run from the repo root: uv run python data/write-up/scripts/confusion_7b_transposed.py

Redraws the 7B confusion matrix with the axes swapped (predicted as rows, actual as columns).
The held-out test activations were not cached, so the cell values are recovered from the colours of the
original 7B figure (docs/images/1c_confusion.png) via its own colour bar; the diagonal uses the exact
percentages printed on that figure. Values are saved to data/write-up/confusion_7b_reconstructed.csv.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg, numpy as np, csv
from matplotlib.colors import ListedColormap

SRC, OUT = "docs/images/1c_confusion.png", "docs/images/1c_confusion_7b.png"
E = ["joyful", "excited", "proud", "calm", "surprised", "sad", "lonely", "ashamed", "afraid", "desperate", "angry", "disgusted"]
DIAG = [.91, .50, .71, .94, .62, .53, .70, .21, .59, .35, .75, .62]   # printed on the original figure

im = mpimg.imread(SRC)[..., :3]
cb_y0, cb_y1, cb_x = 67, 639, 828                                       # colour bar: value 1.0 at y0, 0.0 at y1
cb_cols = im[cb_y0:cb_y1 + 1, cb_x - 3:cb_x + 4].mean(1)
cb_vals = np.linspace(1, 0, len(cb_cols))
x0, x1, y0, y1 = 158, 777, 44, 663                                      # matrix extent in the original
cw, ch = (x1 - x0 + 1) / 12, (y1 - y0 + 1) / 12
R = np.zeros((12, 12))                                                  # R[actual, predicted]
for i in range(12):
    for j in range(12):
        ys, xs = int(y0 + (i + .2) * ch), int(x0 + (j + .2) * cw)       # off-centre, away from the printed text
        c = im[ys:ys + 6, xs:xs + 6].reshape(-1, 3).mean(0)
        R[i, j] = cb_vals[np.argmin(((cb_cols - c) ** 2).sum(1))]
print("max |recovered − printed| on the diagonal:", np.abs(np.diag(R) - DIAG).max().round(3))
np.fill_diagonal(R, DIAG)
R[R < 0.015] = 0
with open("data/write-up/confusion_7b_reconstructed.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["actual \\ predicted"] + E)
    for e, row in zip(E, R): w.writerow([e] + [f"{v:.2f}" for v in row])

cmap = ListedColormap(cb_cols[::-1])                                    # the original's own colour scale
bg, grey = tuple(im[5, 5]), (0.32, 0.32, 0.32)
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(6.2, 5.9), dpi=150, facecolor=bg)
T = R.T                                                                 # rows = predicted, columns = actual
imh = ax.imshow(T, cmap=cmap, vmin=0, vmax=1)
ax.set_xticks(range(12), E, rotation=45, ha="right", fontsize=8, color=grey)
ax.set_yticks(range(12), E, fontsize=8, color=grey)
ax.tick_params(colors=grey, length=3)
for s in ax.spines.values(): s.set_color((0.8, 0.8, 0.8))
ax.set_xlabel("actual", fontsize=9, color=grey)
ax.set_ylabel("predicted", fontsize=9, color=grey, rotation=0, ha="right", va="center")
for k in range(12):
    ax.text(k, k, f"{T[k, k]:.0%}", ha="center", va="center", fontsize=7, color="white" if T[k, k] > 0.55 else "black")
cb = fig.colorbar(imh, ax=ax, fraction=0.04, pad=0.03)
cb.set_label("share of that emotion's stories", fontsize=8, color=grey); cb.ax.tick_params(labelsize=7, colors=grey)
cb.outline.set_edgecolor((0.8, 0.8, 0.8))
fig.suptitle("Confusion Matrix: Linear probe made from emotion directions\nclassifies held out stories with 60% accuracy",
             fontsize=11, fontweight="bold", y=0.975, linespacing=1.3)
fig.text(0.02, 0.02, "Actual vs predicted emotion concepts. A strong diagonal means more accurate,\n"
                     "and off-diagonal cells are misclassifications.", fontsize=7.6, color=grey, ha="left", va="bottom",
         linespacing=1.25)
fig.subplots_adjust(left=0.2, right=0.9, top=0.86, bottom=0.2)
fig.savefig(OUT, dpi=150, facecolor=bg)
print("saved", OUT)
