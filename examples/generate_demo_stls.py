from pathlib import Path


def box(path: Path, x: float, y: float, z: float) -> None:
    vertices = [(0,0,0),(x,0,0),(x,y,0),(0,y,0),(0,0,z),(x,0,z),(x,y,z),(0,y,z)]
    faces = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
             (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    with path.open("w") as f:
        f.write(f"solid {path.stem}\n")
        for face in faces:
            f.write(" facet normal 0 0 0\n  outer loop\n")
            for i in face:
                f.write("   vertex %s %s %s\n" % vertices[i])
            f.write("  endloop\n endfacet\n")
        f.write(f"endsolid {path.stem}\n")


if __name__ == "__main__":
    out = Path(__file__).parent
    for name, dims in [("part_a", (50, 35, 20)), ("part_b", (40, 30, 12)), ("part_c", (25, 20, 15))]:
        box(out / f"{name}.stl", *dims)
