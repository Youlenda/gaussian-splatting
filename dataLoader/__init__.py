# from .llff import LLFFDataset
# from .blender import BlenderDataset
from .blender_entire import BlenderDataset
# from .nsvf import NSVF
# from .tankstemple import TanksTempleDataset
# from .your_own_data import YourOwnDataset



# dataset_dict = {'blender': BlenderDataset,
#                'llff':LLFFDataset,
            #    'tankstemple':TanksTempleDataset,
#                'nsvf':NSVF,
#                 'own_data':YourOwnDataset}
# dataset_dict = {'blender': BlenderDataset,
#                 'tankstemple': TanksTempleDataset}
dataset_dict = {'blender': BlenderDataset}