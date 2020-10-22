# This hook file is needed to package dervet with pyinstaller
from PyInstaller.utils.hooks import copy_metadata

datas = copy_metadata('rainflow')
