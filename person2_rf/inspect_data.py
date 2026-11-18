import os, sys, collections
ROOT = sys.argv[1] if len(sys.argv) > 1 else "Student_folder"
ext_count = collections.Counter()
dirs = collections.defaultdict(list)
for d, _, fs in os.walk(ROOT):
    for f in fs:
        ext_count[os.path.splitext(f)[1]] += 1
        dirs[d].append(f)
print("Extensions:", dict(ext_count))
print("\nFirst 25 dirs:")
for d in list(dirs)[:25]:
    print(d, "->", len(dirs[d]), "files, e.g.", dirs[d][:5])
shown = set()
for d, fs in dirs.items():
    for f in fs:
        e = os.path.splitext(f)[1]
        if e in (".track", ".nodes", ".arcs", ".route") and e not in shown:
            shown.add(e); p = os.path.join(d, f)
            lines = open(p).read().splitlines()
            print(f"\n{p}  ({len(lines)} lines)"); print("\n".join(lines[:4]))