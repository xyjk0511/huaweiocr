# -*- mode: python ; coding: utf-8 -*-
import os
import sys

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

LOCAL_DETECTOR_TARGET = 'local_models\\detectors'
REQUIRED_LOCAL_DETECTORS = (
    os.path.join('local_models', 'detectors', 'label_detector.onnx'),
    os.path.join('local_models', 'detectors', 'field_detector.onnx'),
)
REQUIRED_OCR_MODEL_DIRS = (
    'PP-OCRv5_server_det',
    'PP-OCRv5_server_rec',
    'en_PP-OCRv5_mobile_rec',
    'PP-LCNet_x1_0_textline_ori',
)

missing_local_detectors = [
    path for path in REQUIRED_LOCAL_DETECTORS if not os.path.isfile(path)
]
if missing_local_detectors:
    raise FileNotFoundError(
        "Missing required local ONNX detector model(s) for release build: "
        + ", ".join(missing_local_detectors)
        + ". Place label_detector.onnx and field_detector.onnx under "
        + LOCAL_DETECTOR_TARGET
        + " before running PyInstaller."
    )

missing_ocr_model_dirs = [
    model_dir
    for model_dir in REQUIRED_OCR_MODEL_DIRS
    if not os.path.isdir(os.path.join('bundle', 'models', 'official_models', model_dir))
]
if missing_ocr_model_dirs:
    raise FileNotFoundError(
        "Missing required PaddleOCR model dir(s) for release build: "
        + ", ".join(missing_ocr_model_dirs)
        + ". Expected each under bundle\\models\\official_models\\ before running PyInstaller."
    )

datas = [
    ('bundle\\BarcodeReaderCLI\\bin\\BarcodeReaderCLI.exe', 'BarcodeReaderCLI\\bin'),
    ('bundle\\BarcodeReaderCLI\\bin\\curl.exe', 'BarcodeReaderCLI\\bin'),
    ('bundle\\BarcodeReaderCLI\\bin\\curl-ca-bundle.crt', 'BarcodeReaderCLI\\bin'),
    ('bundle\\BarcodeReaderCLI\\bin\\inlite-barcode-reader-license-agreement.pdf', 'BarcodeReaderCLI\\bin'),
]
datas += [
    (
        os.path.join('bundle', 'models', 'official_models', model_dir),
        os.path.join('models', 'official_models', model_dir),
    )
    for model_dir in REQUIRED_OCR_MODEL_DIRS
]
datas += [
    (path, LOCAL_DETECTOR_TARGET)
    for path in REQUIRED_LOCAL_DETECTORS
]
datas += collect_data_files('paddlex', includes=['.version', 'configs/**'])
_cython_datas = collect_data_files('Cython', includes=['Utility/**'])
if not _cython_datas:
    raise RuntimeError(
        "collect_data_files('Cython', includes=['Utility/**']) returned nothing: the "
        "build venv is missing Cython, so PaddleX runtime files (e.g. "
        "_internal\\Cython\\Utility\\CppSupport.cpp) would be silently omitted and the "
        "packaged app would break at runtime. Install build deps first: "
        "pip install -r requirements.txt (Cython is now pinned there)."
    )
datas += _cython_datas
datas += collect_data_files('tkinterdnd2', includes=['tkdnd/**'])
datas += copy_metadata('opencv-contrib-python')
datas += copy_metadata('pyclipper')
datas += copy_metadata('python-bidi')

binaries = []
binaries += collect_dynamic_libs('paddle')
binaries += collect_dynamic_libs('pyzbar')


a = Analysis(
    ['gui_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=['tkinterdnd2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch',
        'torchvision',
        'playwright',
        'googleapiclient',
        'plotly',
        'pyarrow',
        'polars',
        'duckdb',
        'sklearn',
        'scipy',
        'matplotlib',
        'boto3',
        'botocore',
        'imageio_ffmpeg',
        'llvmlite',
        'numba',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HuaweiOCR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HuaweiOCR',
)
