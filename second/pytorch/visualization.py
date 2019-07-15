import time
import numpy as np
import torch
from google.protobuf import text_format
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import box_coder_builder, input_reader_builder
import pickle
from second.pytorch.visualization_util import showPC


saic_config_path = "../configs/pointpillars/car/xyres_16_saic_prep_and_data_batch_1.proto"

kitti_config_path = "../configs/pointpillars/car/xyres_16_only_for_kitti_analysis.proto"


def visualize_points_and_boxes(data="saic",
                               show_pc=False,
                               save_difficulty_hist=False,
                               show_difficulty_hist=False,
                               show_anchor_assign_hist=False,
                               save_anchor_assign_hist=False,
                               show_gt_filter_out_hist=False):

    if data == "saic":
        config_path = saic_config_path
    elif data == "kitti":
        config_path = kitti_config_path
    else:
        raise ValueError('data not supported!')
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second

    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    ######################
    # BUILD TARGET ASSIGNER
    ######################
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)

    ######################
    # PREPARE INPUT
    ######################
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        dataset=data
    )

    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        dataset=data)

    def _worker_init_fn(worker_id):
        time_seed = np.array(time.time(), dtype=np.int32)
        np.random.seed(time_seed + worker_id)
        print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        collate_fn=merge_second_batch)

    ######################
    # get points, anchors, and ground truth box
    ######################

    data_iter = iter(dataloader)
    assign_hist = np.zeros((1000))
    difficulty_hist = np.zeros((3))
    gt_remove_num = 0
    for i in range(len(dataset)):
        print(i, 'th image')
        try:
            example = next(data_iter)

        except StopIteration:
            print("end epoch")
            break

        gtboxes = example['gt_boxes']
        positive_gt_id = example['positive_gt_id']
        original_gtbox_num = example['original_gtbox_num']
        difficulty = example['gt_difficulties']
        # print("gtboxes num: ", gtboxes.shape[0])
        # print("anchor assigned num", positive_gt_id.shape[1])

        if show_pc is True:
            # show anchor and gt boxes
            anchors = example["anchors"].reshape((-1, 7))
            voxels = example['voxels']
            points = voxels.reshape(-1, 4)[:, :3]
            # print(anchors.shape)  # [-1, 7]
            anchors = anchors.reshape((216, 248, 2, 7))
            anchors = anchors[0::40, 0::40, :, :].reshape((-1, 7))
            # print(anchors[:10, :])  # anchors all have the same lwh, different xyz, theta
            showPC(points, prediction_label=gtboxes, difficulty=difficulty, anchor=anchors)

        if show_anchor_assign_hist:
            assign_info = np.zeros((gtboxes.shape[0]))
            for i in range(gtboxes.shape[0]):
                anchor_assigned_to_this_box = np.sum(positive_gt_id[0, :] == i)
                assign_info[i] = anchor_assigned_to_this_box
                assign_hist[anchor_assigned_to_this_box] += 1
            print(assign_info)

        if show_gt_filter_out_hist:
            original_gtbox_num = original_gtbox_num[0]
            gtboxes_num = gtboxes.shape[0]
            print(gtboxes_num, "/", original_gtbox_num)
            gt_remove_num += original_gtbox_num - gtboxes_num

        if show_difficulty_hist:
            difficulty_hist_temp = np.zeros((3))
            for i in range(3):
                num = np.sum(difficulty[0, :] == i)
                print(num)
                difficulty_hist_temp[i] = num
                difficulty_hist[i] += num
            print('difficulty_hist_temp', difficulty_hist_temp)
        print()

    if save_anchor_assign_hist:
        np.save(data + "anchor_assign_hist" + ".npy", assign_hist)
        print(assign_hist)

    if save_difficulty_hist:
        np.save(data + "difficulty_hist" + ".npy", difficulty_hist)
        print(difficulty_hist)

    print(gt_remove_num)

'''
visualize_points_and_boxes(data='kitti',
                           show_pc=False,
                           show_difficulty_hist=True,
                           save_difficulty_hist=True
                           )
'''

'''
saic_hist = np.load("./saic_hist.npy")
kitti_hist = np.load("./kitti_hist.npy")
print("saic 0 anchor:", saic_hist[0], "  saic 20+ anchors:", np.sum(saic_hist[20:]))

print("kitti 0 anchor:", kitti_hist[0], "  kitti 20+ anchors:", np.sum(kitti_hist[20:]))

plt.plot(saic_hist[:20], 'ro', kitti_hist[:20], 'g^')
plt.xlabel("anchor num")
plt.ylabel("gt box num")
plt.show()
'''


def visualize_from_database_file(db_file):
    with open(info_path, 'rb') as f:
        db_infos = pickle.load(f)

    for k, v in db_infos.iterms():
        print(k)


visualize_from_database_file("/home/yamin/Desktop/sets/kitti_second/kitti_dbinfos_train.pkl")




