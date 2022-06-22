from os import listdir
from os.path import isfile, join
import random

SHAPENET_PATH = "shapenet_dim32_df/"
CLASS_IDS = {
  "airplane": "02691156",
  "cabinet": "02933112",
  "car": "02958343",
  "chair": "03001627",
  "lamp": "03636649",
  "sofa": "04256520",
  "table": "04379243",
  "watercraft": "04530566"
}


def extract1(class1):
    splits = ['train', 'val', 'test']
    file_dir = "splits/shapenet/"
    id1 = CLASS_IDS[class1]
    id1_path = SHAPENET_PATH + id1

    # read files and shuffle file list
    onlyfiles = listdir(id1_path)
    random.shuffle(onlyfiles)

    # get split indicies
    train_index = int(len(onlyfiles) * 0.8)
    val_index = int(len(onlyfiles) * 0.9)
    test_index = int(len(onlyfiles) * 1.0)

    # split data
    onlyfiles_splits = {
        "train": onlyfiles[0:train_index],
        "val": onlyfiles[train_index:val_index],
        "test": onlyfiles[val_index:test_index]
    }

    for split in splits:
        with open(file_dir + class1 + "_" + split + ".txt", 'w') as fp:
            for item in onlyfiles_splits[split]:
                # write each item on a new line
                fp.write(id1 + '/' + item + '\n')
            print(f'Done extracting {class1}_{split} - ({len(onlyfiles_splits[split])})')


def extract2(class1, class2):
    splits = ['train', 'val', 'test']
    file_dir = "splits/shapenet/"
    id1 = CLASS_IDS[class1]
    id2 = CLASS_IDS[class2]
    id1_path = SHAPENET_PATH + id1
    id2_path = SHAPENET_PATH + id2

    # read files and shuffle file list
    onlyfiles1 = listdir(id1_path)
    onlyfiles2 = listdir(id2_path)
    all = onlyfiles1 + onlyfiles2
    random.shuffle(all)

    # get split indicies
    train_index = int(len(all) * 0.8)
    val_index = int(len(all) * 0.9)
    test_index = int(len(all) * 1.0)

    # split data
    onlyfiles_splits = {
        "train": all[0:train_index],
        "val": all[train_index:val_index],
        "test": all[val_index:test_index]
    }

    for split in splits:
        with open(file_dir + class1 + "_" + class2 + "_" + split + ".txt", 'w') as fp:
            for item in onlyfiles_splits[split]:
                # write each item on a new line
                if item in onlyfiles1:
                    fp.write(id1 + '/' + item + '\n')
                elif item in onlyfiles2:
                    fp.write(id2 + '/' + item + '\n')
            print(f'Done extracting {class1}_{class2}_{split} - ({len(onlyfiles_splits[split])})')
