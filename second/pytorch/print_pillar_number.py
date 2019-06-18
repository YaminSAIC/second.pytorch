import time
import numpy as np
import torch
from google.protobuf import text_format
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import box_coder_builder, input_reader_builder
from matplotlib import pyplot as plt
import math



def pillar_number_histogram(config_path):
    config_file_bkp = "pipeline.config"
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    input_cfg = config.train_input_reader
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

    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        dataset='saic'
    )

    def _worker_init_fn(worker_id):
        time_seed = np.array(time.time(), dtype=np.int32)
        np.random.seed(time_seed + worker_id)
        print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size,
        shuffle=True,
        num_workers=input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn)

    data_iter = iter(dataloader)

    pillar_numbers = []

    while 1:
        try:
            example = next(data_iter)
            pillar_numbers.append(example['voxels'].shape[0])

        except StopIteration:
            print("end epoch")
            break

    pillar_numbers = np.array(pillar_numbers)

    bins = np.linspace(min(pillar_numbers),
                       max(pillar_numbers),
                       30)
    plt.hist(pillar_numbers, bins=bins, alpha=0.5)
    plt.title('saic data(fixed bin size)')
    plt.xlabel('variable X (bin size = 30)')
    plt.ylabel('count')
    plt.show()


# pillar_number_histogram("../configs/pointpillars/car/xyres_16_only_for_pillar_num_hist.proto")


