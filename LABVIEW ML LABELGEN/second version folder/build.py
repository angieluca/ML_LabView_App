import PyInstaller.__main__
import os

# Get the absolute path of the current script
script_path = os.path.abspath(__file__)
# Get the directory path of the script
script_dir = os.path.dirname(script_path)
# Construct the path to the model file relative to the script directory
model_path = os.path.join(script_dir, 'best_pleth_ml_model_type2.pkl')

PyInstaller.__main__.run([
    '--name=LabelGenApp',
    '--onefile',
    '--windowed',
    f'--add-data={model_path};.',
    '--hidden-import=openpyxl',
    'labelgen_ml_edu.py'
])