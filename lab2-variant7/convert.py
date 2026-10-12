"""Convert images between the lab RGBA format and PNG."""
import argparse
import pathlib
import struct
from PIL import Image


def convert(mode, source, target):
    if mode == "to-png":
        data = pathlib.Path(source).read_bytes()
        if len(data) < 8:
            raise ValueError("truncated image header")
        width, height = struct.unpack("<ii", data[:8])
        if not 0 < width < 65536 or not 0 < height < 65536 or len(data) != 8 + width * height * 4:
            raise ValueError("invalid dimensions or pixel data")
        image = Image.frombytes("RGBA", (width, height), data[8:])
        image.convert("RGB").save(target, format="PNG")
    elif mode == "to-data":
        with Image.open(source) as image:
            if image.format != "PNG":
                raise ValueError("expected PNG image")
            width, height = image.size
            if not 0 < width < 65536 or not 0 < height < 65536:
                raise ValueError("invalid image size")
            image = image.convert("RGBA")
            image.putalpha(0)
            data = struct.pack("<ii", width, height) + image.tobytes()
        pathlib.Path(target).write_bytes(data)
    else:
        raise ValueError("unknown conversion mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["to-data", "to-png"])
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()
    try:
        convert(args.mode, args.source, args.target)
    except (ValueError, OSError) as error:
        parser.exit(1, f"ERROR: {error}\n")
