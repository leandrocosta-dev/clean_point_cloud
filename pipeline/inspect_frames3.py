import sys

path = sys.argv[1]
size = __import__("os").path.getsize(path)
print("size:", size)

candidates = []
for header_off in range(0, 0x2000, 4):
    rem = size - header_off
    if rem <= 0:
        continue
    for n in range(600, 900):
        if rem % n == 0:
            frame_size = rem // n
            if 5000 <= frame_size <= 400000:
                candidates.append((header_off, n, frame_size))

print("num candidates:", len(candidates))
for c in candidates[:40]:
    print(c)
