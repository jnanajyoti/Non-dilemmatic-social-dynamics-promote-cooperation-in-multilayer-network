"""
Matplotlib helper that draws a legend at the bottom of a figure with the
legend markers at equally spaced horizontal positions.

matplotlib's fig.legend(ncol=N, mode="expand") sizes each column to fit its
content, so the columns do not have equal widths, and trailing whitespace in
a label does not widen a column because matplotlib does not include it in the
text bounding box. draw_uniform_legend therefore draws the legend by hand on
its own axes, with every marker at a fixed x coordinate.

Used by the plot() functions of run_ws_validation_ring_baba.py (Fig G) and
point_b_mc_overlay.py (Fig I), which import it from this folder.

Function
--------
draw_uniform_legend(fig, rows, ncols, ...)   draw the legend; returns its axes

Dependencies: numpy, matplotlib.
"""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D


def draw_uniform_legend(
    fig,
    rows,
    *,
    ncols: int,
    bbox=(0.05, 0.0, 0.9, 0.06),
    fontsize: float = 14,
    marker_size: float = 10,
    handle_text_gap: float = 0.022,   # gap between handle and label, in axes coordinates (0 to 1)
    row_height: float = 0.42,         # vertical distance between rows, in axes coordinates
):
    """Draw a legend with len(rows) rows and ncols columns on a new axes of
    `fig`, with the markers of each row at equally spaced x coordinates.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    rows : list of list of dict
        One list per legend row, each of length `ncols`. Each entry is a dict
        describing one legend item with the keys
          - "kind": "marker" or "line"
          - for "marker":
              "marker": marker style, e.g. "s" or "o"
              "mfc": face colour (default "white")
              "mec": edge colour (default "black")
              "mew": edge width (optional, default 1.4)
          - for "line":
              "color": line colour
              "linestyle": e.g. "--" or "-" (default "-")
              "linewidth": float (default 2.0)
          - "label": label text (matplotlib mathtext allowed)
          - "label_color": text colour (default "black")
    ncols : int
        Number of items per row.
    bbox : (left, bottom, width, height)
        Position of the legend axes in figure coordinates.
    fontsize, marker_size : float
        Label font size and marker size.
    handle_text_gap : float
        Horizontal distance from the centre of a handle to the start of its
        label, in axes coordinates.
    row_height : float
        Vertical distance between the centres of adjacent rows, in axes
        coordinates.

    Returns
    -------
    The matplotlib Axes that holds the legend.
    """
    nrows = len(rows)
    assert all(len(r) == ncols for r in rows), \
        "Every row must have exactly `ncols` items."

    # Axes that holds only the legend: limits 0 to 1, no frame, no ticks.
    lax = fig.add_axes(list(bbox), frameon=False)
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.set_xticks([])
    lax.set_yticks([])
    lax.set_navigate(False)

    # Marker x coordinates: [0, 1] is split into ncols equal slots, and each
    # marker sits 5% of a slot width from the left edge of its slot, so that
    # the label has room to its right.
    slot_width = 1.0 / ncols
    marker_x_centers = np.array([slot_width * (c + 0.05) for c in range(ncols)])

    # Row y coordinates, top row first, centred on y = 0.5 and spaced by
    # row_height.
    if nrows == 1:
        y_centers = np.array([0.5])
    else:
        y_centers = np.linspace(0.5 + (nrows - 1) / 2 * row_height,
                                 0.5 - (nrows - 1) / 2 * row_height,
                                 nrows)

    for r, row in enumerate(rows):
        y = y_centers[r]
        for c, item in enumerate(row):
            x = marker_x_centers[c]
            kind = item["kind"]
            if kind == "marker":
                lax.plot(
                    [x], [y],
                    marker=item["marker"],
                    markersize=marker_size,
                    markerfacecolor=item.get("mfc", "white"),
                    markeredgecolor=item.get("mec", "black"),
                    markeredgewidth=item.get("mew", 1.4),
                    linestyle="None",
                    clip_on=False,
                )
            elif kind == "line":
                # Short horizontal line segment centred at (x, y). Its half
                # length (0.016) is smaller than the default handle_text_gap
                # (0.022), so the line ends before the label starts.
                half = 0.016
                lax.plot(
                    [x - half, x + half], [y, y],
                    color=item["color"],
                    linestyle=item.get("linestyle", "-"),
                    linewidth=item.get("linewidth", 2.0),
                    clip_on=False,
                )
            else:
                raise ValueError(f"unknown kind: {kind!r}")
            # Label, left-aligned at handle_text_gap to the right of the handle.
            lax.text(
                x + handle_text_gap,
                y,
                item["label"],
                fontsize=fontsize,
                va="center", ha="left",
                color=item.get("label_color", "black"),
                clip_on=False,
            )

    return lax
