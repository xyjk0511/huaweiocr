"""Supported input formats and deterministic source-image enumeration."""
import os


SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(folder: str, *, extensions=SUPPORTED_IMAGE_EXTS) -> list[str]:
    images = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames.sort()
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() in extensions:
                images.append(os.path.join(dirpath, name))
    return sorted(images, key=lambda p: os.path.normcase(os.path.relpath(p, folder)))
