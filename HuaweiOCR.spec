# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import copy_metadata

datas = [('bundle\\models', 'models'), ('bundle\\BarcodeReaderCLI', 'BarcodeReaderCLI'), ('C:\\Users\\55093\\PycharmProjects\\PythonProject24\\.venv\\Lib\\site-packages\\paddlex\\.version', 'paddlex'), ('C:\\Users\\55093\\PycharmProjects\\PythonProject24\\.venv\\Lib\\site-packages\\paddlex\\configs', 'paddlex\\configs')]
datas += copy_metadata('opencv-contrib-python')
datas += copy_metadata('pyclipper')
datas += copy_metadata('python-bidi')


a = Analysis(
    ['gui_app.py'],
    pathex=[],
    binaries=[
        ('C:\\Users\\55093\\PycharmProjects\\PythonProject24\\.venv\\Lib\\site-packages\\paddle\\libs\\*', 'paddle\\libs'),
        ('C:\\Users\\55093\\PycharmProjects\\PythonProject24\\.venv\\Lib\\site-packages\\pyzbar\\libiconv.dll', 'pyzbar'),
        ('C:\\Users\\55093\\PycharmProjects\\PythonProject24\\.venv\\Lib\\site-packages\\pyzbar\\libzbar-64.dll', 'pyzbar'),
        ('C:\\Windows\\System32\\vcruntime140.dll', '.'),
        ('C:\\Windows\\System32\\vcruntime140_1.dll', '.'),
        ('C:\\Windows\\System32\\msvcp140.dll', '.'),
        ('C:\\Windows\\System32\\msvcp140_1.dll', '.'),
        ('C:\\Windows\\System32\\msvcp140_2.dll', '.'),
        ('C:\\Windows\\System32\\concrt140.dll', '.'),
        ('C:\\Windows\\System32\\vcomp140.dll', '.'),
    ],
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
