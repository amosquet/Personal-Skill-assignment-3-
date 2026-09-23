#!/usr/bin/env python3
"""Pack STL meshes onto one virtual printer plate and export a combined STL.

This is a conservative, axis-aligned first-fit row packer. It does not rotate
models or replace a production slicer's collision/support analysis.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Mesh:
    path: Path
    triangles: list[tuple[tuple[float, float, float], ...]]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(self.bounds_max[i] - self.bounds_min[i] for i in range(3))


@dataclass
class Placement:
    filename: str
    x: float
    y: float
    z: float
    width: float
    depth: float
    height: float


def _mesh_from_triangles(path: Path, triangles: list[tuple[tuple[float, float, float], ...]]) -> Mesh:
    if not triangles:
        raise ValueError(f"{path}: STL contains no triangles")
    vertices = [v for tri in triangles for v in tri]
    if not all(math.isfinite(c) for v in vertices for c in v):
        raise ValueError(f"{path}: STL contains a non-finite coordinate")
    mins = tuple(min(v[i] for v in vertices) for i in range(3))
    maxs = tuple(max(v[i] for v in vertices) for i in range(3))
    if any(maxs[i] - mins[i] <= 0 for i in range(3)):
        raise ValueError(f"{path}: model has zero size on at least one axis")
    return Mesh(path, triangles, mins, maxs)


def load_stl(path: Path) -> Mesh:
    """Load binary or ASCII STL using only the Python standard library."""
    data = path.read_bytes()
    # Binary STL is exactly 84 bytes plus 50 bytes per triangle. Checking the
    # size avoids misclassifying binary files whose header starts with 'solid'.
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if 84 + count * 50 == len(data):
            tris = []
            offset = 84
            for _ in range(count):
                values = struct.unpack_from("<12fH", data, offset)
                verts = tuple(tuple(float(c) for c in values[3 + j * 3:6 + j * 3]) for j in range(3))
                tris.append(verts)
                offset += 50
            return _mesh_from_triangles(path, tris)

    text = data.decode("utf-8", errors="replace")
    points = [tuple(float(n) for n in m) for m in re.findall(
        r"\bvertex\s+([-+\deE.]+)\s+([-+\deE.]+)\s+([-+\deE.]+)", text, re.I
    )]
    if len(points) % 3:
        raise ValueError(f"{path}: malformed ASCII STL; vertex count is not divisible by 3")
    tris = [tuple(points[i:i + 3]) for i in range(0, len(points), 3)]
    return _mesh_from_triangles(path, tris)


def pack(meshes: list[Mesh], bed_x: float, bed_y: float, bed_z: float, gap: float) -> list[Placement]:
    """Pack models in descending footprint area using simple row placement."""
    if not meshes:
        raise ValueError("Provide at least one STL file")
    if min(bed_x, bed_y, bed_z) <= 0 or gap < 0:
        raise ValueError("Bed dimensions must be positive and clearance cannot be negative")

    items = sorted(meshes, key=lambda m: (m.size[0] * m.size[1], max(m.size[:2])), reverse=True)
    x = gap
    y = gap
    row_depth = 0.0
    placements = []
    for mesh in items:
        width, depth, height = mesh.size
        label = mesh.path.name
        if width + 2 * gap > bed_x or depth + 2 * gap > bed_y or height > bed_z:
            raise ValueError(f"{label} ({width:.2f} x {depth:.2f} x {height:.2f} mm) cannot fit the bed with {gap:g} mm clearance")
        if x + width + gap > bed_x:
            x = gap
            y += row_depth + gap
            row_depth = 0.0
        if y + depth + gap > bed_y:
            raise ValueError(f"Batch does not fit a {bed_x:g} x {bed_y:g} mm bed with {gap:g} mm clearance")
        placements.append(Placement(label, x, y, 0.0, width, depth, height))
        x += width + gap
        row_depth = max(row_depth, depth)
    return placements


def export_combined_stl(meshes: list[Mesh], placements: list[Placement], output: Path) -> None:
    placement_by_name = {p.filename: p for p in placements}
    # Names can collide across folders; preserve ordering by zipping the exact
    # packing order with a filename lookup only where filenames are unique.
    if len(placement_by_name) != len(placements):
        raise ValueError("Input STL filenames must be unique")
    mesh_by_name = {m.path.name: m for m in meshes}
    if len(mesh_by_name) != len(meshes):
        raise ValueError("Input STL filenames must be unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = sum(len(m.triangles) for m in meshes)
    with output.open("wb") as f:
        f.write(b"Hive batch plate prototype".ljust(80, b"\0"))
        f.write(struct.pack("<I", count))
        for placement in placements:
            mesh = mesh_by_name[placement.filename]
            for tri in mesh.triangles:
                shifted = [(
                    v[0] - mesh.bounds_min[0] + placement.x,
                    v[1] - mesh.bounds_min[1] + placement.y,
                    v[2] - mesh.bounds_min[2] + placement.z,
                ) for v in tri]
                a, b, c = shifted
                u = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
                v = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
                n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
                length = math.sqrt(sum(q*q for q in n)) or 1.0
                normal = tuple(q / length for q in n)
                f.write(struct.pack("<12fH", *normal, *shifted[0], *shifted[1], *shifted[2], 0))


def run_prusaslicer(executable: str, stl: Path, output: Path, profile: Path | None) -> None:
    """Run PrusaSlicer CLI; profile should be a compatible exported .ini."""
    cmd = [executable, "--export-gcode", "--output", str(output)]
    if profile:
        cmd += ["--load", str(profile)]
    cmd.append(str(stl))
    subprocess.run(cmd, check=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stl", nargs="+", type=Path, help="STL files to combine")
    parser.add_argument("-o", "--output", type=Path, default=Path("build/batch_plate.stl"))
    parser.add_argument("--bed-x", type=float, default=256.0, help="usable bed width in mm")
    parser.add_argument("--bed-y", type=float, default=256.0, help="usable bed depth in mm")
    parser.add_argument("--bed-z", type=float, default=256.0, help="usable bed height in mm")
    parser.add_argument("--gap", type=float, default=5.0, help="minimum XY clearance in mm")
    parser.add_argument("--report", type=Path, help="write placement report as JSON")
    parser.add_argument("--prusaslicer", help="optional PrusaSlicer executable path")
    parser.add_argument("--profile", type=Path, help="optional PrusaSlicer .ini profile")
    parser.add_argument("--gcode", type=Path, default=Path("build/batch_plate.gcode"))
    args = parser.parse_args(argv)

    try:
        meshes = [load_stl(p) for p in args.stl]
        placements = pack(meshes, args.bed_x, args.bed_y, args.bed_z, args.gap)
        export_combined_stl(meshes, placements, args.output)
        report = {
            "bed_mm": {"x": args.bed_x, "y": args.bed_y, "z": args.bed_z},
            "clearance_mm": args.gap,
            "input_count": len(meshes),
            "triangle_count": sum(len(m.triangles) for m in meshes),
            "combined_stl": str(args.output),
            "placements": [asdict(p) for p in placements],
            "limitations": ["Axis-aligned row packing only; no rotation or optimal nesting.", "A slicer must still check supports, material, machine profile, and printability."],
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if args.prusaslicer:
            run_prusaslicer(args.prusaslicer, args.output, args.gcode, args.profile)
            print(f"Sliced G-code: {args.gcode}")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
