"""Generate the vector workflow and mesh figures for the revised manuscript."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "figures"
INK = "#193344"
BLUE = "#31678b"
ORANGE = "#a25129"
TEAL = "#276c60"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 7.7})


def drawing(width, height):
    fig = plt.figure(figsize=(width/25.4, height/25.4))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, label, color=BLUE, fill="#f3f7fa"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                lw=.65, ec=color, fc=fill))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", color=INK, linespacing=1.2)


def arrow(ax, start, end):
    ax.annotate("", end, start, arrowprops={"arrowstyle": "-|>", "lw": .7,
                                           "color": INK, "shrinkA": 0, "shrinkB": 0})


def workflow():
    fig, ax = drawing(153, 76)
    box(ax, 2, 59, 33, 13, "Noisy tones\nand chirps")
    box(ax, 42, 59, 31, 13, "One-sided\npower spectrum")
    box(ax, 82, 59, 69, 13, "Conventional spectral controls\nPeak-bin, three-bin and log spectrum", TEAL, "#f2f8f6")
    arrow(ax, (35, 65.5), (42, 65.5))
    arrow(ax, (73, 65.5), (82, 65.5))
    box(ax, 42, 40, 69, 12, "Four descriptors\nStandardize, then fit PCA")
    arrow(ax, (57.5, 59), (57.5, 52))
    ax.plot([76.5, 76.5], [40, 34], color=INK, lw=.7)
    ax.plot([25, 128], [34, 34], color=INK, lw=.7)
    for x in (25, 76.5, 128):
        arrow(ax, (x, 34), (x, 27))
    box(ax, 2, 3, 46, 24, "Classical controls\n2 or 4 PCs\nLinear and nonlinear models")
    box(ax, 53.5, 3, 46, 24, "Photonic classifier\n4 PCs → signed amplitudes\n6 MZIs → differential readout", TEAL, "#f2f8f6")
    box(ax, 105, 3, 46, 24, "Quantum classifiers\n2 or 4 PCs → ZZ encoding\nVariational rotations + CX\nParity decision", ORANGE, "#faf5f1")
    fig.savefig(OUT/"figure1_revised_workflow.pdf")
    plt.close(fig)


def mesh():
    fig, ax = drawing(153, 54)
    ys = [37, 28, 19, 10]
    cols = [39, 53, 67, 81]
    ax.text(13, 47, "Signed input", ha="center", color=INK)
    ax.text(63, 47, "MZI columns", ha="center", color=INK)
    ax.text(101, 47, "Detectors", ha="center", color=INK)
    ax.text(137, 47, "Electronic readout", ha="center", color=INK)
    for index, y in enumerate(ys):
        ax.text(11.5, y, f"$x_{index}$", ha="center", va="center", color=INK)
        arrow(ax, (18, y), (93, y))
        box(ax, 93, y-2.8, 15, 5.6, f"$I_{index}$")
        ax.plot([108, 115], [y, y], color=INK, lw=.65)
    for column, (x, pairs) in enumerate(zip(cols, [[(0, 1), (2, 3)], [(1, 2)],
                                                    [(0, 1), (2, 3)], [(1, 2)]]), 1):
        ax.text(x, 41.7, str(column), ha="center", color=INK)
        for a, b in pairs:
            ax.add_patch(Rectangle((x-3, ys[b]-1.2), 6, ys[a]-ys[b]+2.4,
                                   lw=.7, ec=TEAL, fc="#e2f0ec"))
            ax.plot([x-1.8, x+1.8], [ys[a]-.9, ys[b]+.9], color=TEAL, lw=.7)
            ax.plot([x-1.8, x+1.8], [ys[b]+.9, ys[a]-.9], color=TEAL, lw=.7)
    for top, bottom in [(37, 28), (19, 10)]:
        center = (top+bottom)/2
        ax.plot([115, 115], [top, bottom], color=INK, lw=.65)
        arrow(ax, (115, center), (123, center))
    box(ax, 123, 8, 28, 31, "$I_0+I_1$\n\n$g\\Delta I+b$\n\n$I_2+I_3$", ORANGE, "#faf5f1")
    ax.text(75, 2.8, "$\\Delta I=(I_2+I_3)-(I_0+I_1)$", ha="center", color=INK)
    fig.savefig(OUT/"figure_mesh.pdf")
    plt.close(fig)


if __name__ == "__main__":
    workflow()
    mesh()
