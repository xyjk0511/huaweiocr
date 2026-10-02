"""Supported input formats and deterministic source-image enumeration."""
import os


SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def write_image_unicode(path, img) -> bool:
    """Encode in memory so Windows also writes non-ASCII paths correctly."""
    import cv2

    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        buf.tofile(path)
    except OSError:
        return False
    return os.path.exists(path) and os.path.getsize(path) > 0


def list_images(folder: str, *, extensions=SUPPORTED_IMAGE_EXTS) -> list[str]:
    images = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames.sort()
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() in extensions:
                images.append(os.path.join(dirpath, name))
    return sorted(images, key=lambda p: os.path.normcase(os.path.relpath(p, folder)))
