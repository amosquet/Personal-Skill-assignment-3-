import struct
import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batchplate import export_combined_stl, load_stl, pack


def box_stl(path: Path, x: float, y: float, z: float) -> Path:
    # Twelve triangles for a rectangular prism.
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


if __name__ == "__main__":
    unittest.main()
