import struct
import sys

path = sys.argv[1]
vf_count = int(sys.argv[2]) if len(sys.argv) > 2 else 788

with open(path, "rb") as f:
    data = f.read()

size = len(data)
print(f"file size: {size}, vf_count: {vf_count}")

# try a range of header offsets and uint16/uint32 table interpretations
for dtype, dsize, fmt in [("uint16", 2, "<H"), ("uint32", 4, "<I")]:
    for header_off in range(0, 0x200, 4):
        table_bytes = vf_count * dsize
        if header_off + table_bytes > size:
            continue
        table = struct.unpack(f"<{vf_count}{fmt[1]}", data[header_off:header_off + table_bytes])
        total = sum(table)
        remaining = size - (header_off + table_bytes)
        if total == remaining and all(t > 0 for t in table):
            print(f"MATCH dtype={dtype} header_off={hex(header_off)} sum(table)={total} remaining={remaining}")
            print("first 10 entries:", table[:10])
            print("min/max entries:", min(table), max(table))

print("done scanning exact matches")

# also just print stats of candidate table at 0x60 for both widths, unfiltered
for dtype, dsize, fmt in [("uint16", 2, "H"), ("uint32", 4, "I")]:
    off = 0x60
    table_bytes = vf_count * dsize
    if off + table_bytes <= size:
        table = struct.unpack(f"<{vf_count}{fmt}", data[off:off + table_bytes])
        print(f"\n[{dtype} @ {hex(off)}] first 20:", table[:20])
        print(f"[{dtype} @ {hex(off)}] sum:", sum(table), " remaining_if_header_ends_here:", size - (off + table_bytes))
