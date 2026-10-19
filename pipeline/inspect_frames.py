import sys

path = sys.argv[1]
magic = b"RPAlgo_file_sys"

with open(path, "rb") as f:
    data = f.read()

print(f"file size: {len(data)}")

positions = []
start = 0
while True:
    idx = data.find(magic, start)
    if idx == -1:
        break
    positions.append(idx)
    start = idx + 1

print(f"magic occurrences: {len(positions)}")
print("first 20 positions:", positions[:20])
if len(positions) > 1:
    diffs = [positions[i + 1] - positions[i] for i in range(min(50, len(positions) - 1))]
    print("first 50 diffs (stride between magics):", diffs)
    import collections
    all_diffs = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    cnt = collections.Counter(all_diffs)
    print("most common diffs:", cnt.most_common(10))
