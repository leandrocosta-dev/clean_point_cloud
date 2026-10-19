import sys
import struct

path = sys.argv[1]
header_off = 116
frame_size = 149639
n = 788

with open(path, "rb") as f:
    data = f.read()

one_f32 = struct.pack("<f", 1.0)  # 00 00 80 3f
print("looking for float32 1.0 pattern in frame0 and frame1")

for fi in range(3):
    off = header_off + fi * frame_size
    frame = data[off:off + frame_size]
    positions = []
    start = 0
    while True:
        idx = frame.find(one_f32, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    print(f"frame {fi}: {len(positions)} occurrences of 1.0f, first 20 offsets:", positions[:20])

# also check for identity-like 4x4 (16 floats) pattern: 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
identity = b"".join(struct.pack("<f", v) for v in [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])
idx = data.find(identity)
print("identity 4x4 found at global offset:", idx)

# dump the non-zero region boundaries in frame0 precisely
off0 = header_off
frame0 = data[off0:off0 + frame_size]
nz = [i for i, b in enumerate(frame0) if b != 0]
if nz:
    print("frame0 nonzero byte range:", nz[0], "to", nz[-1], "count_nonzero:", len(nz), "/", frame_size)
