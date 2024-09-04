import os

#ROOT_PATH = os.path.join(os.path.dirname(__file__), 'saved_data')
ROOT_PATH = 'E:/tactile_msc-main'

# Model Paths
TIP_ID = '331'
BASE_DATA_PATH = os.path.join(ROOT_PATH, f"collect_{TIP_ID}_5D_surface")
BASE_MODEL_PATH = os.path.join(ROOT_PATH, f"collect_{TIP_ID}_5D_surface", "model")

SAVED_MODEL_PATH = os.path.join('E:/tactile_msc-main/models') # For transfer learning
