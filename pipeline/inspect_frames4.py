import struct
import sys

path = sys.argv[1]
header_off = 116
frame_size = 149639
n = 788

with open(path, "rb") as f:
    data = f.read()

print("total size:", len(data))
print("header bytes (0:116):")
print(data[:116].hex())

frame0 = data[header_off:header_off + frame_size]
print("\nframe0 first 64 bytes hex:", frame0[:64].hex())
print("frame0 last 64 bytes hex:", frame0[-64:].hex())

frame1_off = header_off + frame_size
frame1 = data[frame1_off:frame1_off + frame_size]
print("\nframe1 first 64 bytes hex:", frame1[:64].hex())

# try interpreting frame0 as float32 array after small headers of various sizes
import numpy as np
for h in [0, 4, 8, 11, 12, 16, 20, 23, 24, 32]:
    body = frame0[h:]
    usable = (len(body) // 4) * 4
    arr = np.frombuffer(body[:usable], dtype="<f4")
    finite = np.isfinite(arr)
    frac_finite = finite.mean()
    if frac_finite > 0.9:
        sub = arr[finite]
        print(f"h={h}: n_floats={len(arr)} frac_finite={frac_finite:.3f} min={sub.min():.3f} max={sub.max():.3f} mean_abs={np.abs(sub).mean():.3f}")
