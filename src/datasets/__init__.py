from .SyntheticDataset import SyntheticDataset
from .CenterDirGroundtruthDataset import CenterDirGroundtruthDataset
from .LockableSeedRandomAccess import LockableSeedRandomAccess
from .SorghumPlantCentersDataset import SorghumPlantCentersDataset
from .CARPKandPUCPRplusDataset import CARPKandPUCPRplusDataset
from .TreeCountingDataset import TreeCountingDataset
from models.center_groundtruth import CenterDirGroundtruth

def get_raw_dataset(name, dataset_opts):
    if name == "sorghum":
        dataset = SorghumPlantCentersDataset(**dataset_opts)
    elif name.lower() == 'carpk':
        dataset = CARPKandPUCPRplusDataset(db_name='CARPK', **dataset_opts)
    elif name.lower() == 'pucpr+':
        dataset = CARPKandPUCPRplusDataset(db_name='PUCPR+', **dataset_opts)
    elif name.lower() == 'acacia_06':
        dataset = TreeCountingDataset(name='Acacia_06', **dataset_opts)
    elif name.lower() == 'acacia_12':
        dataset = TreeCountingDataset(name='Acacia_12', **dataset_opts)
    elif name.lower() == 'oilpalm':
        dataset = TreeCountingDataset(name='Oilpalm', **dataset_opts)
    elif name == "syn":
        dataset = SyntheticDataset(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))


    return dataset

def get_dataset(name, dataset_opts, centerdir_gt_opts=None):
    dataset = get_raw_dataset(name, dataset_opts)
    if centerdir_gt_opts is not None and len(centerdir_gt_opts) > 0:
        centerdir_groundtruth_op = CenterDirGroundtruth(**centerdir_gt_opts)

        dataset = CenterDirGroundtruthDataset(dataset, centerdir_groundtruth_op)
        return dataset, centerdir_groundtruth_op
    else:
        return dataset, None


