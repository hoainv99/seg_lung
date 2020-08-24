import os
import cv2
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix  

def get_data_dict(data_path,label_path):
    folder = os.listdir(label_path)
    data_dicts = []
    for id_folder in folder:
        try:
            file = os.listdir(os.path.join(label_path,id_folder))
            for name in file:
                label_name = os.path.join(id_folder,name)
                image_name = os.path.join(id_folder,name[:-3]+'jpg')
                data_dicts.append({
                    'image_path' : os.path.join(data_path,image_name),
                    'label_path' : os.path.join(label_path,label_name)
                })

        except:
            continue
    return data_dicts


def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='path to training dataset')
    parser.add_argument('--label_path', required=True, help='path to validation dataset')
    opt = parser.parse_args()
    dict = get_data_dict(opt.data_path,opt.label_path)
    print(len(dict))