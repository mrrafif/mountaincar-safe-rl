"""Tile several per-run training videos into one synchronized comparison grid.

Each input is a ``training.mp4`` produced by the recorder during training (it
already carries its own header with the hyperparameters used). This script lays
them out in a grid, advancing all of them in lock-step. When a shorter video
ends, its cell freezes on the final frame until the longest one finishes.

Examples
--------
Compare three PPO seeds of one config::

    python scripts/make_grid_video.py \
        --videos results/ppo/seed_0/training.mp4 \
                 results/ppo/seed_1/training.mp4 \
                 results/ppo/seed_2/training.mp4 \
        --cols 3 --out videos/ppo_seeds_grid.mp4

Compare every run found under results/ppo (one cell per run)::

    python scripts/make_grid_video.py \
        --glob "results/ppo/**/training.mp4" --cols 3 \
        --out videos/ppo_all_grid.mp4
"""
from __future__ import annotations

import argparse
import glob as globlib
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

_BORDER = 3
_BG = (24, 24, 27)


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return arr.astype(np.uint8)


def _scaled(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    img = Image.fromarray(frame).resize((int(w * scale), int(h * scale)))
    return np.asarray(img)


def _assemble(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    """Lay tiles out in a `cols`-wide grid, padding the remainder with bg."""
    n = len(tiles)
    rows = (n + cols - 1) // cols
    ch, cw = tiles[0].shape[:2]
    canvas = np.zeros(
        (rows * ch + (rows + 1) * _BORDER, cols * cw + (cols + 1) * _BORDER, 3),
        dtype=np.uint8,
    )
    canvas[:] = _BG
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        y = _BORDER + r * (ch + _BORDER)
        x = _BORDER + c * (cw + _BORDER)
        canvas[y:y + ch, x:x + cw] = tile
    return canvas


def build_grid(files: list[str], out: str, cols: int = 3, scale: float = 0.5,
               fps: int = 30, verbose: bool = True) -> int:
    """Tile per-run videos into one synchronized grid. Returns frame count.

    Shorter videos freeze on their final frame until the longest finishes.
    Callable directly from other scripts (e.g. run_stage1.py).
    """
    files = [f for f in files if Path(f).exists()]
    if not files:
        raise FileNotFoundError("build_grid: no existing input videos given")
    if verbose:
        print(f"Tiling {len(files)} videos into a {cols}-column grid -> {out}")
        for f in files:
            print("  -", f)

    readers = [imageio.get_reader(f) for f in files]
    iters = [iter(r) for r in readers]
    last: list[np.ndarray | None] = [None] * len(files)
    exhausted = [False] * len(files)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=fps, macro_block_size=8, codec="libx264")

    n_out = 0
    while not all(exhausted):
        tiles = []
        for i, it in enumerate(iters):
            if not exhausted[i]:
                try:
                    frame = _scaled(_to_rgb(next(it)), scale)
                    last[i] = frame
                except StopIteration:
                    exhausted[i] = True
                    frame = last[i]
            else:
                frame = last[i]
            tiles.append(frame)
        writer.append_data(_assemble(tiles, cols))
        n_out += 1

    writer.close()
    for r in readers:
        r.close()
    if verbose:
        print(f"Wrote {out}  ({n_out} frames)")
    return n_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=str, nargs="*", default=[],
                        help="explicit list of training.mp4 paths")
    parser.add_argument("--glob", type=str, default=None,
                        help="glob pattern (relative to repo root) for training.mp4 files")
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--scale", type=float, default=0.5,
                        help="per-cell downscale factor (0.5 keeps files small)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    paths: list[str] = list(args.videos)
    if args.glob:
        paths += sorted(str(p) for p in ROOT.glob(args.glob))
        paths += sorted(globlib.glob(args.glob, recursive=True))
    # de-dup while preserving order, keep only existing files
    seen, files = set(), []
    for p in paths:
        rp = str(Path(p))
        if rp not in seen and Path(rp).exists():
            seen.add(rp)
            files.append(rp)

    if not files:
        sys.exit("No input videos found. Pass --videos or --glob.")

    build_grid(files, args.out, cols=args.cols, scale=args.scale, fps=args.fps)


if __name__ == "__main__":
    main()
