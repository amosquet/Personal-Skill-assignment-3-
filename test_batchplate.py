import json
import struct
import tempfile
import unittest
from unittest.mock import patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batchplate import (
    IncompatibleBatchError,
    PrintJob,
    build_orcaslicer_command,
    export_combined_stl,
    group_compatible_jobs,
    load_jobs_from_manifest,
    load_stl,
    main,
    pack,
    run_slicer,
    validate_batch_compatibility,
)


def box_stl(path: Path, x: float, y: float, z: float) -> Path:
    # Twelve triangles for a rectangular prism.
    path.parent.mkdir(parents=True, exist_ok=True)
    p = [(0,0,0),(x,0,0),(x,y,0),(0,y,0),(0,0,z),(x,0,z),(x,y,z),(0,y,z)]
    faces = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
             (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    with path.open("w") as f:
        f.write("solid box\n")
        for face in faces:
            f.write(" facet normal 0 0 0\n  outer loop\n")
            for index in face:
                f.write("   vertex %s %s %s\n" % p[index])
            f.write("  endloop\n endfacet\n")
        f.write("endsolid box\n")
    return path


class BatchPlateTests(unittest.TestCase):
    def test_ascii_stl_dimensions_and_three_part_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            meshes = [load_stl(box_stl(root / f"part{i}.stl", 20, 10, 5)) for i in range(3)]
            self.assertEqual(meshes[0].size, (20.0, 10.0, 5.0))
            places = pack(meshes, 100, 100, 50, 5)
            self.assertEqual(len(places), 3)
            self.assertEqual([p.x for p in places], [5, 30, 55])
            self.assertTrue(all(p.y == 5 for p in places))

    def test_rejects_batch_that_exceeds_bed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            meshes = [load_stl(box_stl(root / f"part{i}.stl", 30, 10, 5)) for i in range(3)]
            with self.assertRaisesRegex(ValueError, "does not fit"):
                pack(meshes, 70, 30, 50, 5)

    def test_export_is_binary_stl_with_all_triangles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mesh = load_stl(box_stl(root / "part.stl", 20, 10, 5))
            placements = pack([mesh], 100, 100, 50, 5)
            out = root / "combined.stl"
            export_combined_stl([mesh], placements, out)
            data = out.read_bytes()
            self.assertEqual(struct.unpack_from("<I", data, 80)[0], 12)
            self.assertEqual(len(data), 84 + 12 * 50)
            self.assertAlmostEqual(struct.unpack_from("<f", data, 84 + 12)[0], 5.0)

    # ------------------------------------------------------------------------
    # 1. Slicer Alignment (Orca Slicer CLI) Tests
    # ------------------------------------------------------------------------
    def test_build_orcaslicer_command(self):
        cmd = build_orcaslicer_command(
            executable="/usr/bin/orca-slicer",
            stl=Path("build/plate.stl"),
            output_gcode=Path("build/plate.gcode"),
            machine_profile=Path("profiles/bambu_h2d.json"),
            process_profile=Path("profiles/0.20mm_standard.json"),
            filament_profile=Path("profiles/pla_basic.json"),
        )
        self.assertEqual(
            cmd,
            [
                "/usr/bin/orca-slicer",
                "--slice",
                "0",
                "--outputdir",
                "build",
                "--load-settings",
                "profiles/bambu_h2d.json;profiles/0.20mm_standard.json",
                "--load-filaments",
                "profiles/pla_basic.json",
                "build/plate.stl",
            ],
        )

    def test_build_orcaslicer_command_with_3mf_export(self):
        cmd = build_orcaslicer_command(
            executable="orca-slicer",
            stl=Path("plate.stl"),
            output_gcode=Path("out/plate.gcode.3mf"),
            profile=Path("preset.json"),
        )
        self.assertIn("--export-3mf", cmd)
        self.assertIn("out/plate.gcode.3mf", cmd)

    @patch("batchplate.subprocess.run")
    def test_run_slicer_dispatch_orca_and_prusa(self, mock_run):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stl = root / "plate.stl"
            gcode = root / "out" / "plate.gcode"

            run_slicer(
                engine="orca",
                executable="/usr/bin/orca-slicer",
                stl=stl,
                output_gcode=gcode,
                machine_profile=Path("voron.json"),
                filament_profile=Path("abs.json"),
            )
            mock_run.assert_called_once()
            orca_cmd = mock_run.call_args[0][0]
            self.assertEqual(orca_cmd[:3], ["/usr/bin/orca-slicer", "--slice", "0"])
            self.assertIn("--load-settings", orca_cmd)
            self.assertIn("--load-filaments", orca_cmd)

            mock_run.reset_mock()
            run_slicer(
                engine="prusa",
                executable="/usr/bin/prusa-slicer",
                stl=stl,
                output_gcode=gcode,
                profile=Path("prusa.ini"),
            )
            mock_run.assert_called_once()
            prusa_cmd = mock_run.call_args[0][0]
            self.assertEqual(
                prusa_cmd,
                ["/usr/bin/prusa-slicer", "--export-gcode", "--output", str(gcode), "--load", "prusa.ini", str(stl)],
            )

            with self.assertRaisesRegex(ValueError, "Unknown slicer engine"):
                run_slicer("cura", "/usr/bin/cura", stl, gcode)

    # ------------------------------------------------------------------------
    # 3. Multi-User & Material Compatibility Tests
    # ------------------------------------------------------------------------
    def test_material_compatibility_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job1 = PrintJob.from_stl(
                box_stl(root / "user1.stl", 15, 15, 5),
                job_id="job-101",
                user_id="asabale@purdue.edu",
                material="PLA",
                nozzle_diameter=0.4,
            )
            job2 = PrintJob.from_stl(
                box_stl(root / "user2.stl", 20, 20, 5),
                job_id="job-102",
                user_id="dhillon3@purdue.edu",
                material="pla",
                nozzle_diameter=0.4,
            )
            validate_batch_compatibility([job1, job2])
            placements = pack([job1, job2], 100, 100, 50, 5)
            self.assertEqual(len(placements), 2)

            place_by_job = {p.job_id: p for p in placements}
            self.assertEqual(place_by_job["job-101"].user_id, "asabale@purdue.edu")
            self.assertEqual(place_by_job["job-101"].material, "PLA")
            self.assertEqual(place_by_job["job-102"].user_id, "dhillon3@purdue.edu")

    def test_rejects_incompatible_materials(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job1 = PrintJob.from_stl(
                box_stl(root / "p1.stl", 10, 10, 5),
                user_id="student1@purdue.edu",
                material="PLA",
            )
            job2 = PrintJob.from_stl(
                box_stl(root / "p2.stl", 10, 10, 5),
                user_id="student2@purdue.edu",
                material="PETG",
            )
            with self.assertRaises(IncompatibleBatchError) as ctx:
                validate_batch_compatibility([job1, job2])
            self.assertIn("conflicting materials", str(ctx.exception).lower())

            with self.assertRaises(IncompatibleBatchError):
                pack([job1, job2], 100, 100, 50, 5)

    def test_rejects_different_nozzle_diameters_and_profiles_and_temps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stl1 = box_stl(root / "p1.stl", 10, 10, 5)
            stl2 = box_stl(root / "p2.stl", 10, 10, 5)

            # Nozzle diameter mismatch
            j1 = PrintJob.from_stl(stl1, material="PLA", nozzle_diameter=0.4)
            j2 = PrintJob.from_stl(stl2, material="PLA", nozzle_diameter=0.6)
            with self.assertRaisesRegex(IncompatibleBatchError, "nozzle diameters"):
                validate_batch_compatibility([j1, j2])

            # Print profile mismatch
            j3 = PrintJob.from_stl(stl1, material="PLA", print_profile="0.12mm Fine")
            j4 = PrintJob.from_stl(stl2, material="PLA", print_profile="0.28mm Draft")
            with self.assertRaisesRegex(IncompatibleBatchError, "print profiles"):
                validate_batch_compatibility([j3, j4])

            # Bed temperature mismatch
            j5 = PrintJob.from_stl(stl1, material="PLA", bed_temperature=60.0)
            j6 = PrintJob.from_stl(stl2, material="PLA", bed_temperature=75.0)
            with self.assertRaisesRegex(IncompatibleBatchError, "bed temperatures"):
                validate_batch_compatibility([j5, j6])

    def test_multi_user_same_filename_disambiguated_by_job_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Two students upload files both named 'bracket.stl' in their own directories
            j1 = PrintJob.from_stl(
                box_stl(root / "u1" / "bracket.stl", 15, 10, 5),
                job_id="job-u1",
                user_id="amosquet@purdue.edu",
                material="PLA",
            )
            j2 = PrintJob.from_stl(
                box_stl(root / "u2" / "bracket.stl", 20, 10, 5),
                job_id="job-u2",
                user_id="awasthi2@purdue.edu",
                material="PLA",
            )
            placements = pack([j1, j2], 100, 100, 50, 5)
            out = root / "combined.stl"
            export_combined_stl([j1, j2], placements, out)
            data = out.read_bytes()
            self.assertEqual(struct.unpack_from("<I", data, 80)[0], 24)

    def test_group_compatible_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            j_pla_1 = PrintJob.from_stl(box_stl(root / "a.stl", 10, 10, 5), material="PLA", nozzle_diameter=0.4)
            j_pla_2 = PrintJob.from_stl(box_stl(root / "b.stl", 10, 10, 5), material="PLA", nozzle_diameter=0.4)
            j_petg = PrintJob.from_stl(box_stl(root / "c.stl", 10, 10, 5), material="PETG", nozzle_diameter=0.4)
            j_pla_large_nozzle = PrintJob.from_stl(box_stl(root / "d.stl", 10, 10, 5), material="PLA", nozzle_diameter=0.8)

            groups = group_compatible_jobs([j_pla_1, j_pla_2, j_petg, j_pla_large_nozzle])
            self.assertEqual(len(groups), 3)
            self.assertEqual(len(groups[("PLA", 0.4)]), 2)
            self.assertEqual(len(groups[("PETG", 0.4)]), 1)
            self.assertEqual(len(groups[("PLA", 0.8)]), 1)

    def test_load_manifest_and_cli_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p1 = box_stl(root / "part1.stl", 15, 10, 5)
            p2 = box_stl(root / "part2.stl", 20, 15, 5)
            manifest = [
                {
                    "path": str(p1),
                    "job_id": "job-1",
                    "user_id": "amosquet@purdue.edu",
                    "material": "PLA",
                    "nozzle_diameter": 0.4,
                },
                {
                    "path": str(p2),
                    "job_id": "job-2",
                    "user_id": "awasthi2@purdue.edu",
                    "material": "PLA",
                    "nozzle_diameter": 0.4,
                },
            ]
            m_path = root / "manifest.json"
            m_path.write_text(json.dumps(manifest))

            out_stl = root / "out.stl"
            report_path = root / "report.json"
            rc = main(["--manifest", str(m_path), "-o", str(out_stl), "--report", str(report_path)])
            self.assertEqual(rc, 0)
            self.assertTrue(out_stl.exists())
            report_data = json.loads(report_path.read_text())
            self.assertEqual(report_data["batch_manifest"]["material"], "PLA")
            self.assertEqual(
                report_data["batch_manifest"]["users"],
                ["amosquet@purdue.edu", "awasthi2@purdue.edu"],
            )


if __name__ == "__main__":
    unittest.main()
