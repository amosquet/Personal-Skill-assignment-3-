#!/usr/bin/env python3
"""PrintHive BatchPlate: Pack STL meshes onto one virtual build plate and export a combined STL.

Integrates multi-user job tracking, material compatibility validation,
and headless Orca Slicer CLI pipeline support.
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
from typing import Iterable, Sequence


class IncompatibleBatchError(ValueError):
    """Raised when models to be packed on the same build plate have incompatible print parameters."""
    pass


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
class PrintJob:
    """Print job representation with multi-user attribution and print specifications."""
    mesh: Mesh
    job_id: str = ""
    user_id: str = ""
    material: str = "PLA"
    print_profile: str = "Standard"
    bed_temperature: float | None = None
    nozzle_temperature: float | None = None
    nozzle_diameter: float = 0.4

    def __post_init__(self) -> None:
        self.job_id = self.job_id.strip() or self.mesh.path.stem
        self.user_id = self.user_id.strip() or "anonymous@purdue.edu"
        self.material = self.material.strip().upper()
        self.print_profile = self.print_profile.strip() or "Standard"
        if not self.material:
            raise ValueError(f"{self.mesh.path}: material cannot be empty")
        if self.nozzle_diameter <= 0 or not math.isfinite(self.nozzle_diameter):
            raise ValueError(f"{self.mesh.path}: nozzle_diameter must be a positive number")

    @classmethod
    def from_stl(
        cls,
        path: Path,
        job_id: str = "",
        user_id: str = "",
        material: str = "PLA",
        print_profile: str = "Standard",
        bed_temperature: float | None = None,
        nozzle_temperature: float | None = None,
        nozzle_diameter: float = 0.4,
    ) -> PrintJob:
        mesh = load_stl(path)
        return cls(
            mesh=mesh,
            job_id=job_id,
            user_id=user_id,
            material=material,
            print_profile=print_profile,
            bed_temperature=bed_temperature,
            nozzle_temperature=nozzle_temperature,
            nozzle_diameter=nozzle_diameter,
        )


@dataclass
class Placement:
    filename: str
    x: float
    y: float
    z: float
    width: float
    depth: float
    height: float
    job_id: str = ""
    user_id: str = ""
    material: str = "PLA"


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


def validate_batch_compatibility(jobs: Sequence[PrintJob]) -> None:
    """Validate that all print jobs in a batch share compatible material and print settings."""
    if not jobs:
        return

    materials = {j.material.strip().upper() for j in jobs}
    if len(materials) > 1:
        details = ", ".join(f"{j.mesh.path.name} (user: {j.user_id}): {j.material}" for j in jobs)
        raise IncompatibleBatchError(
            f"Cannot batch models with conflicting materials on one plate: {sorted(materials)}. Details: {details}"
        )

    nozzle_diameters = {j.nozzle_diameter for j in jobs}
    if len(nozzle_diameters) > 1:
        raise IncompatibleBatchError(
            f"Cannot batch models requiring different nozzle diameters: {sorted(nozzle_diameters)}"
        )

    profiles = {j.print_profile.strip() for j in jobs if j.print_profile and j.print_profile.strip()}
    if len({p.lower() for p in profiles}) > 1:
        raise IncompatibleBatchError(
            f"Cannot batch models with conflicting print profiles on one plate: {sorted(profiles)}"
        )

    bed_temps = {j.bed_temperature for j in jobs if j.bed_temperature is not None}
    if len(bed_temps) > 1:
        raise IncompatibleBatchError(
            f"Conflicting bed temperatures requested in batch: {sorted(bed_temps)}"
        )

    nozzle_temps = {j.nozzle_temperature for j in jobs if j.nozzle_temperature is not None}
    if len(nozzle_temps) > 1:
        raise IncompatibleBatchError(
            f"Conflicting nozzle temperatures requested in batch: {sorted(nozzle_temps)}"
        )


def group_compatible_jobs(jobs: Sequence[PrintJob]) -> dict[tuple[str, float], list[PrintJob]]:
    """Group heterogeneous print queue submissions into compatible batches by (material, nozzle_diameter)."""
    groups: dict[tuple[str, float], list[PrintJob]] = {}
    for job in jobs:
        key = (job.material.strip().upper(), job.nozzle_diameter)
        groups.setdefault(key, []).append(job)
    return groups


def pack(
    items: Sequence[Mesh | PrintJob],
    bed_x: float,
    bed_y: float,
    bed_z: float,
    gap: float,
    validate_compatibility: bool = True,
) -> list[Placement]:
    """Pack models in descending footprint area using simple row placement.

    Accepts either Mesh or PrintJob objects. If PrintJob objects are provided,
    verifies that material and print settings match across the batch.
    """
    if not items:
        raise ValueError("Provide at least one STL file")
    if min(bed_x, bed_y, bed_z) <= 0 or gap < 0:
        raise ValueError("Bed dimensions must be positive and clearance cannot be negative")

    # If PrintJobs are provided, run compatibility checks
    print_jobs = [item for item in items if isinstance(item, PrintJob)]
    if print_jobs and validate_compatibility:
        if len(print_jobs) != len(items):
            raise TypeError("Cannot mix raw Mesh and PrintJob instances in a validated batch")
        validate_batch_compatibility(print_jobs)

    def _get_mesh(it: Mesh | PrintJob) -> Mesh:
        return it.mesh if isinstance(it, PrintJob) else it

    def _get_job_info(it: Mesh | PrintJob) -> tuple[str, str, str]:
        if isinstance(it, PrintJob):
            return it.job_id, it.user_id, it.material
        return it.path.stem, "", "PLA"

    sorted_items = sorted(
        items,
        key=lambda it: (
            _get_mesh(it).size[0] * _get_mesh(it).size[1],
            max(_get_mesh(it).size[:2]),
        ),
        reverse=True,
    )

    x = gap
    y = gap
    row_depth = 0.0
    placements = []
    for item in sorted_items:
        mesh = _get_mesh(item)
        job_id, user_id, material = _get_job_info(item)
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
        placements.append(
            Placement(
                filename=label,
                x=x,
                y=y,
                z=0.0,
                width=width,
                depth=depth,
                height=height,
                job_id=job_id,
                user_id=user_id,
                material=material,
            )
        )
        x += width + gap
        row_depth = max(row_depth, depth)
    return placements


def export_combined_stl(
    items: Sequence[Mesh | PrintJob],
    placements: list[Placement],
    output: Path,
) -> None:
    meshes = [it.mesh if isinstance(it, PrintJob) else it for it in items]
    all_print_jobs = all(isinstance(it, PrintJob) for it in items)

    # Match placements to meshes by unique job_id when PrintJobs have distinct job_ids
    # (allowing different students to submit files with the same basename),
    # otherwise fall back to unique filename lookup.
    mesh_lookup: dict[str, Mesh] = {}
    use_job_id = False
    if all_print_jobs:
        job_ids = [it.job_id for it in items if isinstance(it, PrintJob)]
        placement_job_ids = [p.job_id for p in placements if p.job_id]
        if len(set(job_ids)) == len(items) and len(set(placement_job_ids)) == len(placements):
            mesh_lookup = {it.job_id: it.mesh for it in items if isinstance(it, PrintJob)}
            use_job_id = True

    if not use_job_id:
        placement_by_name = {p.filename: p for p in placements}
        if len(placement_by_name) != len(placements):
            raise ValueError("Input STL filenames must be unique")
        mesh_lookup = {m.path.name: m for m in meshes}
        if len(mesh_lookup) != len(meshes):
            raise ValueError("Input STL filenames must be unique")

    output.parent.mkdir(parents=True, exist_ok=True)
    count = sum(len(m.triangles) for m in meshes)
    with output.open("wb") as f:
        f.write(b"PrintHive batch plate".ljust(80, b"\0"))
        f.write(struct.pack("<I", count))
        for placement in placements:
            key = placement.job_id if use_job_id else placement.filename
            mesh = mesh_lookup[key]
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


# ============================================================================
# Slicer Integration (Orca Slicer CLI & PrusaSlicer)
# ============================================================================

def build_orcaslicer_command(
    executable: str | Path,
    stl: Path,
    output_gcode: Path,
    profile: Path | None = None,
    machine_profile: Path | None = None,
    filament_profile: Path | None = None,
    process_profile: Path | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Construct command-line arguments for headless Orca Slicer CLI.

    Orca Slicer CLI uses:
      --slice 0
      --outputdir <dir>
      --load-settings "<machine.json>;<process.json>"
      --load-filaments "<filament.json>"
    """
    output_dir = output_gcode.parent if str(output_gcode.parent) else Path(".")
    cmd = [str(executable), "--slice", "0", "--outputdir", str(output_dir)]

    settings = [str(p) for p in (profile, machine_profile, process_profile) if p is not None]
    if settings:
        cmd += ["--load-settings", ";".join(settings)]

    if filament_profile is not None:
        cmd += ["--load-filaments", str(filament_profile)]

    if output_gcode.suffix.lower() == ".3mf":
        cmd += ["--export-3mf", str(output_gcode)]

    if extra_args:
        cmd.extend(extra_args)

    cmd.append(str(stl))
    return cmd


def run_orcaslicer(
    executable: str | Path,
    stl: Path,
    output_gcode: Path,
    profile: Path | None = None,
    machine_profile: Path | None = None,
    filament_profile: Path | None = None,
    process_profile: Path | None = None,
    extra_args: list[str] | None = None,
) -> None:
    """Run Orca Slicer CLI in headless mode to slice model into G-code."""
    output_gcode.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_orcaslicer_command(
        executable=executable,
        stl=stl,
        output_gcode=output_gcode,
        profile=profile,
        machine_profile=machine_profile,
        filament_profile=filament_profile,
        process_profile=process_profile,
        extra_args=extra_args,
    )
    subprocess.run(cmd, check=True)


def run_prusaslicer(executable: str, stl: Path, output: Path, profile: Path | None) -> None:
    """Run PrusaSlicer CLI; profile should be a compatible exported .ini."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [executable, "--export-gcode", "--output", str(output)]
    if profile:
        cmd += ["--load", str(profile)]
    cmd.append(str(stl))
    subprocess.run(cmd, check=True)


def run_slicer(
    engine: str,
    executable: str | Path,
    stl: Path,
    output_gcode: Path,
    profile: Path | None = None,
    machine_profile: Path | None = None,
    filament_profile: Path | None = None,
    process_profile: Path | None = None,
) -> None:
    """Run slicer abstraction dispatching to Orca Slicer or PrusaSlicer."""
    engine_lower = engine.lower().strip()
    if engine_lower in ("orca", "orcaslicer", "orca-slicer"):
        run_orcaslicer(
            executable=executable,
            stl=stl,
            output_gcode=output_gcode,
            profile=profile,
            machine_profile=machine_profile,
            filament_profile=filament_profile,
            process_profile=process_profile,
        )
    elif engine_lower in ("prusa", "prusaslicer", "prusa-slicer"):
        run_prusaslicer(str(executable), stl, output_gcode, profile)
    else:
        raise ValueError(f"Unknown slicer engine: '{engine}'. Expected 'orca' or 'prusa'.")


def load_jobs_from_manifest(manifest_path: Path) -> list[PrintJob]:
    """Load PrintJob definitions from a JSON manifest file."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Manifest must be a non-empty JSON array of job objects: {manifest_path}")

    jobs: list[PrintJob] = []
    base_dir = manifest_path.parent
    for idx, item in enumerate(data):
        if not isinstance(item, dict) or "path" not in item:
            raise ValueError(f"Manifest entry {idx} must be an object containing a 'path' field")
        raw_path = Path(item["path"])
        resolved_path = raw_path if raw_path.is_absolute() else base_dir / raw_path
        jobs.append(
            PrintJob.from_stl(
                path=resolved_path,
                job_id=str(item.get("job_id", "")),
                user_id=str(item.get("user_id", "anonymous@purdue.edu")),
                material=str(item.get("material", "PLA")),
                print_profile=str(item.get("print_profile", "Standard")),
                bed_temperature=float(item["bed_temperature"]) if item.get("bed_temperature") is not None else None,
                nozzle_temperature=float(item["nozzle_temperature"]) if item.get("nozzle_temperature") is not None else None,
                nozzle_diameter=float(item.get("nozzle_diameter", 0.4)),
            )
        )
    return jobs


# ============================================================================
# CLI Entrypoint
# ============================================================================

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stl", nargs="*", type=Path, help="STL files to combine (or use --manifest)")
    parser.add_argument("-o", "--output", type=Path, default=Path("build/batch_plate.stl"))
    parser.add_argument("--manifest", type=Path, help="JSON manifest containing multi-user job definitions")
    parser.add_argument("--user-id", default="anonymous@purdue.edu", help="default user ID for positional STL inputs")
    parser.add_argument("--material", default="PLA", help="default material for positional STL inputs (e.g. PLA, PETG, ABS)")
    parser.add_argument("--print-profile", default="Standard", help="default print profile for positional STL inputs")
    parser.add_argument("--nozzle-diameter", type=float, default=0.4, help="nozzle diameter in mm (default: 0.4)")
    parser.add_argument("--bed-x", type=float, default=256.0, help="usable bed width in mm (default: 256.0)")
    parser.add_argument("--bed-y", type=float, default=256.0, help="usable bed depth in mm (default: 256.0)")
    parser.add_argument("--bed-z", type=float, default=256.0, help="usable bed height in mm (default: 256.0)")
    parser.add_argument("--gap", type=float, default=5.0, help="minimum XY clearance in mm (default: 5.0)")
    parser.add_argument("--report", type=Path, help="write placement report as JSON")

    # Slicer configuration
    parser.add_argument("--slicer", choices=["orca", "prusa"], default="orca", help="slicing engine (default: orca)")
    parser.add_argument("--orcaslicer", help="path to Orca Slicer CLI executable")
    parser.add_argument("--prusaslicer", help="path to PrusaSlicer executable (legacy)")
    parser.add_argument("--profile", type=Path, help="general slicer profile (.ini or .json)")
    parser.add_argument("--machine-profile", type=Path, help="Orca Slicer machine profile (e.g. Bambu, Prusa, Voron)")
    parser.add_argument("--filament-profile", type=Path, help="Orca Slicer filament profile")
    parser.add_argument("--process-profile", type=Path, help="Orca Slicer process profile")
    parser.add_argument("--gcode", type=Path, default=Path("build/batch_plate.gcode"))
    args = parser.parse_args(argv)

    try:
        if args.manifest:
            jobs = load_jobs_from_manifest(args.manifest)
            if args.stl:
                for p in args.stl:
                    jobs.append(
                        PrintJob.from_stl(
                            p,
                            user_id=args.user_id,
                            material=args.material,
                            print_profile=args.print_profile,
                            nozzle_diameter=args.nozzle_diameter,
                        )
                    )
        elif args.stl:
            jobs = [
                PrintJob.from_stl(
                    p,
                    user_id=args.user_id,
                    material=args.material,
                    print_profile=args.print_profile,
                    nozzle_diameter=args.nozzle_diameter,
                )
                for p in args.stl
            ]
        else:
            parser.error("Must provide STL files or a --manifest JSON file")

        # Validate multi-user batch compatibility & pack
        validate_batch_compatibility(jobs)
        placements = pack(jobs, args.bed_x, args.bed_y, args.bed_z, args.gap)
        export_combined_stl(jobs, placements, args.output)

        report = {
            "bed_mm": {"x": args.bed_x, "y": args.bed_y, "z": args.bed_z},
            "clearance_mm": args.gap,
            "batch_manifest": {
                "material": jobs[0].material,
                "print_profile": jobs[0].print_profile,
                "nozzle_diameter": jobs[0].nozzle_diameter,
                "users": sorted({j.user_id for j in jobs}),
                "job_count": len(jobs),
                "job_ids": [j.job_id for j in jobs],
            },
            "input_count": len(jobs),
            "triangle_count": sum(len(j.mesh.triangles) for j in jobs),
            "combined_stl": str(args.output),
            "placements": [asdict(p) for p in placements],
            "limitations": [
                "Axis-aligned row packing only; no rotation or optimal nesting.",
                "A slicer must still check supports, material, machine profile, and printability.",
            ],
        }

        slicer_bin = args.orcaslicer or args.prusaslicer
        if slicer_bin:
            engine = "prusa" if args.prusaslicer and not args.orcaslicer else args.slicer
            report["slicer"] = {
                "engine": engine,
                "executable": slicer_bin,
                "gcode": str(args.gcode),
            }

        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))

        if slicer_bin:
            engine = "prusa" if args.prusaslicer and not args.orcaslicer else args.slicer
            run_slicer(
                engine=engine,
                executable=slicer_bin,
                stl=args.output,
                output_gcode=args.gcode,
                profile=args.profile,
                machine_profile=args.machine_profile,
                filament_profile=args.filament_profile,
                process_profile=args.process_profile,
            )
            print(f"Sliced G-code: {args.gcode}")
        return 0

    except (OSError, ValueError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
