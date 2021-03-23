# from utils import create_data_lists
from utils import create_userDef_data_lists
import os

if __name__ == '__main__':
    # create_data_lists(voc07_path='merge2007/VOC2007',
    #                   voc12_path='VOCtrainval_11-May-2012/VOCdevkit/VOC2012',
    #                   output_folder='./')
    cwd = os.getcwd()
    path = os.path.abspath(cwd)
    dataset_path = os.path.join(path, 'DataSet')
    create_userDef_data_lists(userDefDataset_path=dataset_path,
                              output_folder=path)
