from .kitti_common import *
from second.core import preprocess as prep
import numpy as np


# put all the bounding boxes to easy
def add_difficulty_to_annos_saic(info):
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    diff = []
    for i in range(len(dims)):
        diff.append(0)
    annos["difficulty"] = np.array(diff, np.int32)
    return diff


def get_label_anno_saic(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])

    #annotations['truncated'] = np.array([float(x[1]) for x in content])
    #annotations['occluded'] = np.array([int(x[2]) for x in content])

    annotations['alpha'] = np.array([float(x[3]) for x in content])

    # bbox can not be deleted
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
            -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def get_label_anno_saic_within_range(label_path,
                                     bv_range):
    annotations = get_label_anno_saic(label_path)
    gtbox = np.concatenate([annotations['location'], annotations['dimensions'], annotations['rotation_y']], axis=1)
    range_filter = prep.filter_gt_box_outside_range(gtbox, bv_range)
    
    for key in ["name","alpha","bbox","dimensions","location","rotation_y","score","index","group_ids"]:
        if annotations[key] is not None:
            annotations[key] = annotations[key][range_filter]
    return annotations


def get_saic_image_info(path,
                        training=True,
                        label_info=True,
                        velodyne=False,
                        image_ids=7481,
                        num_worker=8,
                        relative_path=True):
    root_path = pathlib.Path(path)
    if not isinstance(image_ids, list):
        raise ValueError("image_ids has to be a list since "
                         "the ids are not consecutive.")

    def map_func(idx):
        image_info = {'image_idx': idx, 'pointcloud_num_features': 4}
        annotations = None
        if velodyne:
            image_info['velodyne_path'] = get_velodyne_path(
                idx, path, training, relative_path)

        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            # actually the label is exactly the same as kitti
            # the only difference is: unused params are set to 0
            annotations = get_label_anno_saic(label_path)
        if annotations is not None:
            image_info['annos'] = annotations
            add_difficulty_to_annos_saic(image_info)
        return image_info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)
    return list(image_infos)


def saic_box_cam_to_second_box_lidar(gt_boxes_camera):
    x, y, z = gt_boxes_camera[:, 0:1], gt_boxes_camera[:, 1:2], gt_boxes_camera[:, 2:3]
    l, h, w = gt_boxes_camera[:, 3:4], gt_boxes_camera[:, 4:5], gt_boxes_camera[:, 5:6]
    r = gt_boxes_camera[:, 6:7]
    return np.concatenate([z, -x, -y, w, l, h, r], axis=1)


def saic_lidar_points_to_second_lidar_points(saic_lidar_points):
    x, y, z, i = saic_lidar_points[:, 0:1], saic_lidar_points[:, 1:2], \
                 saic_lidar_points[:, 2:3], saic_lidar_points[:, 3:4]
    return np.concatenate([y, -x, z, i], axis=1)

"""
# wrong now
def second_box_lidar_to_saic_box_cam(second_pred_boxes_lidar):
    x, y, z = second_pred_boxes_lidar[:, 0:1], second_pred_boxes_lidar[:, 1:2], second_pred_boxes_lidar[:, 2:3]
    w, l, h = second_pred_boxes_lidar[:, 3:4], second_pred_boxes_lidar[:, 4:5], second_pred_boxes_lidar[:, 5:6]
    r = second_pred_boxes_lidar[6:7]
    return np.concatenate([-y, z, x, l, h, w, r])
"""
