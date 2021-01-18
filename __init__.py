__all__ = ['dervet', 'storagevet']

import os
import sys

# ADD DERVET and STORAGEVET TO PYTHONPATH
dervet_path = os.path.abspath(__file__)
# add dervet (source root) to PYTHONPATH
sys.path.insert(0, dervet_path)
# dervet's directory path is the first in sys.path

# determine storagevet path (absolute path)
storagevet_path = os.path.join(dervet_path, 'storagevet')
# add storagevet (source root) to PYTHONPATH
sys.path.insert(0, storagevet_path)