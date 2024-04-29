import PyInstaller.__main__

PyInstaller.__main__.run([
    '--name=FileExplorerApp',
    '--onefile',
    '--windowed',
    'labelgen_ml_edu.py'
])