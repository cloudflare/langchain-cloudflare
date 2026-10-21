"""PNG fixtures shared by chat and native extraction vision tests."""

# MARK: - IMPORTS
import base64


# MARK: - IMAGE FIXTURES
def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Create a PNG chunk."""
    import struct
    import zlib

    chunk = chunk_type + data
    return (
        struct.pack(">I", len(data))
        + chunk
        + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    )


def create_test_image_base64() -> str:
    """Create a minimal solid-red PNG and return as base64.

    Uses raw PNG bytes to avoid requiring PIL in the test environment.
    16x16, not 1x1: some vision models (e.g. @cf/google/gemma-4-26b-a4b-it)
    reject images smaller than 10x10px with a 400 ("image dimensions must be
    at least 10px"), so a 1x1 fixture image broke vision tests for those
    models specifically.
    """
    import struct
    import zlib

    width, height = 16, 16
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw_row = b"\x00" + b"\xff\x00\x00" * width
    idat_data = zlib.compress(raw_row * height)

    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", ihdr_data)
    png += _png_chunk(b"IDAT", idat_data)
    png += _png_chunk(b"IEND", b"")

    return base64.standard_b64encode(png).decode("utf-8")


_BLOCK_FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    ":": ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


def _draw_block_text(
    pixels: list,
    width: int,
    x: int,
    y: int,
    text: str,
    scale: int = 7,
) -> None:
    """Draw uppercase block-font text into an RGB pixel buffer."""
    height = len(pixels) // width
    for char in text:
        glyph = _BLOCK_FONT.get(char, _BLOCK_FONT[" "])
        for row, pattern in enumerate(glyph):
            for col, bit in enumerate(pattern):
                if bit != "1":
                    continue
                for dy in range(scale):
                    for dx in range(scale):
                        px = x + col * scale + dx
                        py = y + row * scale + dy
                        if 0 <= px < width and 0 <= py < height:
                            pixels[py * width + px] = (0, 0, 0)
        x += 6 * scale


def create_labeled_test_image_base64() -> str:
    """Create a PNG with large text labels for vision extraction tests."""
    import struct
    import zlib

    width, height = 700, 260
    pixels = [(255, 255, 255)] * (width * height)
    _draw_block_text(pixels, width, 30, 35, "TICKER: ACME")
    _draw_block_text(pixels, width, 30, 115, "TIMEFRAME: Q4")
    _draw_block_text(pixels, width, 30, 195, "DATE: 2026-06-21")

    raw = b"".join(
        b"\x00"
        + b"".join(bytes(pixel) for pixel in pixels[y * width : (y + 1) * width])
        for y in range(height)
    )
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw))
    png += _png_chunk(b"IEND", b"")

    return base64.standard_b64encode(png).decode("utf-8")
