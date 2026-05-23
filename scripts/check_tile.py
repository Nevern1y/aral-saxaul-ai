import rasterio, sys
f = sys.argv[1]
with rasterio.open(f) as src:
    h, w = src.height, src.width
    print(f"{f}: {src.count}x{h}x{w}")
    for y in [9000, 9984, 10240, 11008]:
        wh = min(256, h - y)
        win = rasterio.windows.Window(0, y, w, wh)
        d = src.read(1, window=win)
        v = int((d != -32767).sum())
        print(f"  Y={y}: {wh}x{w}, valid={v}/{wh*w}")
