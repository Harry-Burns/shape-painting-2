import numpy as np
import cv2


def load_img(imgfile, greyscale=True, size: tuple[int, int] | None = None):
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE if greyscale else cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(imgfile)

    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    return img

def display(x: np.ndarray):
        # If float in [0,1], convert to uint8 [0,255]
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 1) if x.max() <= 1.0 else np.clip(x, 0, 255)
            x = (x * 255).astype(np.uint8) if x.max() <= 1 else x.astype(np.uint8)

        # If RGB -> BGR for OpenCV
        if x.ndim == 3 and x.shape[2] == 3:
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

        cv2.imshow("ShapePainter", x)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def display_n_with_stats(
    imgs: list[np.ndarray],
    pad: int = 5,
    wait: int = 0,
    title: str = "ShapePainter",
    best_fitness: float | None = None,
    fitness_hist: list[float] | None = None,
    out_w: int = 600,
    plot_h: int = 140,
):
    if not imgs:
        return

    # ---- normalise/convert to BGR ----
    proc = []
    for x in imgs:
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 1) if x.max() <= 1.0 else np.clip(x, 0, 255)
            x = (x * 255).astype(np.uint8) if x.max() <= 1 else x.astype(np.uint8)

        if x.ndim == 2:
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        elif x.ndim == 3 and x.shape[2] == 3:
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

        proc.append(x)

    # ---- match heights (before horizontal stack) ----
    h = max(im.shape[0] for im in proc)
    resized = [
        cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_NEAREST)
        if im.shape[0] != h else im
        for im in proc
    ]

    # ---- stack side-by-side with padding (image strip) ----
    if pad > 0:
        pad_col = np.zeros((h, pad, 3), dtype=np.uint8)
        out = []
        for i, im in enumerate(resized):
            out.append(im)
            if i < len(resized) - 1:
                out.append(pad_col)
        strip = np.hstack(out)
    else:
        strip = np.hstack(resized)

    # ---- resize strip to fixed width, keep aspect ratio ----
    Hs, Ws = strip.shape[:2]
    if Ws != out_w:
        s = out_w / Ws
        new_h = max(1, int(round(Hs * s)))
        strip = cv2.resize(strip, (out_w, new_h), interpolation=cv2.INTER_AREA)

    # ---- append plot panel UNDER the resized strip ----
    canvas = np.vstack([strip, np.zeros((plot_h, out_w, 3), dtype=np.uint8)])

    H, W = canvas.shape[:2]
    plot_y0 = H - plot_h

    # ---- overlay current best text (on the strip) ----
    if best_fitness is not None:
        cv2.putText(
            canvas,
            f"best fitness: {best_fitness:.6f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # ---- plot fitness history ----
    if fitness_hist:
        hist_all = np.asarray(fitness_hist, dtype=np.float32)
        N_all = hist_all.size

        max_points = min(600, N_all)
        hist = hist_all[-max_points:]
        Np = hist.size

        mn = float(hist.min())
        mx = float(hist.max())
        if mx <= mn:
            mx = mn + 1e-6

        # plot area (within the plot panel)
        margin_l, margin_r, margin_t, margin_b = 60, 15, 15, 35
        x0 = margin_l
        x1 = W - margin_r
        y0 = plot_y0 + margin_t
        y1 = H - margin_b

        # background + border
        canvas[y0:y1, x0:x1] = 0
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (200, 200, 200), 1)

        def to_px(i, v):
            px = x0 + int(i * (x1 - x0) / max(1, Np - 1))
            t = (v - mn) / (mx - mn)
            py = y1 - int(t * (y1 - y0))
            return px, py

        # line
        pts = np.empty((Np, 1, 2), dtype=np.int32)
        for i in range(Np):
            pts[i, 0] = to_px(i, float(hist[i]))
        cv2.polylines(canvas, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)

        # Y ticks (min/mid/max)
        for frac, val in [(0.0, mn), (0.5, (mn + mx) * 0.5), (1.0, mx)]:
            yy = y1 - int(frac * (y1 - y0))
            cv2.line(canvas, (x0 - 5, yy), (x0, yy), (200, 200, 200), 1)
            cv2.putText(
                canvas,
                f"{val:.3f}",
                (8, yy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        # X ticks: gen first/mid/last
        gen_last = N_all - 1
        gen_first = max(0, N_all - max_points)
        gen_mid = (gen_first + gen_last) // 2

        for gx, g in [(x0, gen_first), ((x0 + x1) // 2, gen_mid), (x1, gen_last)]:
            cv2.line(canvas, (gx, y1), (gx, y1 + 5), (200, 200, 200), 1)
            txt = str(g)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(
                canvas,
                txt,
                (gx - tw // 2, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        # axis labels
        cv2.putText(canvas, "Fitness", (x0, plot_y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Gen", (W // 2 - 12, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow(title, canvas)
    key = cv2.waitKey(wait) & 0xFF
    if wait == 0:
        cv2.destroyAllWindows()
    return key





def display_n_grid(
    imgs: list[np.ndarray],
    pad: int = 5,
    wait: int = 0,
    title: str = "ShapePainter",
    cols: int | None = None,     # None -> auto (roughly square)
    max_w: int | None = None,    # e.g. 1600
    max_h: int | None = None,    # e.g. 900
):
    if not imgs:
        return

    # ---- normalise/convert to BGR ----
    proc = []
    for x in imgs:
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 1) if x.max() <= 1.0 else np.clip(x, 0, 255)
            x = (x * 255).astype(np.uint8) if x.max() <= 1 else x.astype(np.uint8)

        if x.ndim == 2:
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        elif x.ndim == 3 and x.shape[2] == 3:
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

        proc.append(x)

    # ---- match heights ----
    h = max(im.shape[0] for im in proc)
    resized = [
        cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_NEAREST)
        if im.shape[0] != h else im
        for im in proc
    ]

    n = len(resized)
    if cols is None:
        cols = int(np.ceil(np.sqrt(n)))
    cols = max(1, cols)
    rows = int(np.ceil(n / cols))

    # pad missing tiles with black
    tile_h = h
    tile_w = max(im.shape[1] for im in resized)
    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    tiles = [
        cv2.copyMakeBorder(im, 0, 0, 0, tile_w - im.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for im in resized
    ]
    tiles += [blank] * (rows * cols - n)

    # build grid
    pad_col = np.zeros((tile_h, pad, 3), dtype=np.uint8) if pad > 0 else None
    pad_row = np.zeros((pad, cols * tile_w + (cols - 1) * pad, 3), dtype=np.uint8) if pad > 0 else None

    grid_rows = []
    for r in range(rows):
        row_tiles = tiles[r * cols : (r + 1) * cols]
        if pad > 0:
            row = np.hstack([t if i == cols - 1 else np.hstack([t, pad_col]) for i, t in enumerate(row_tiles)])
        else:
            row = np.hstack(row_tiles)
        grid_rows.append(row)

    if pad > 0:
        canvas = np.vstack([rr if i == rows - 1 else np.vstack([rr, pad_row]) for i, rr in enumerate(grid_rows)])
    else:
        canvas = np.vstack(grid_rows)

    # ---- final resize if too large ----
    if max_w is not None or max_h is not None:
        H, W = canvas.shape[:2]
        sx = (max_w / W) if max_w else 1.0
        sy = (max_h / H) if max_h else 1.0
        s = min(sx, sy, 1.0)
        if s < 1.0:
            canvas = cv2.resize(canvas, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)

    cv2.imshow(title, canvas)
    key = cv2.waitKey(wait) & 0xFF
    if wait == 0:
        cv2.destroyAllWindows()
    return key
