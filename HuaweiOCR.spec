# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

datas = [
    ('bundle\\models', 'models'),
    ('bundle\\BarcodeReaderCLI\\bin\\BarcodeReaderCLI.exe', 'BarcodeReaderCLI\\bin'),
    ('bundle\\BarcodeReaderCLI\\bin\\curl.exe', 'BarcodeReaderCLI\\bin'),
    ('bundle\\BarcodeReaderCLI\\bin\\curl-ca-bundle.crt', 'BarcodeReaderCLI\\bin'),
    ('bundle\\BarcodeReaderCLI\\bin\\inlite-barcode-reader-license-agreement.pdf', 'BarcodeReaderCLI\\bin'),
]
datas += collect_data_files('paddlex', includes=['.version', 'configs/**'])
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
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
