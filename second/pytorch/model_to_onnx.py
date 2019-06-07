import pathlib
import torch
from google.protobuf import text_format
from second.protos import pipeline_pb2
from second.builder import target_assigner_builder, voxel_builder
from second.pytorch.builder import box_coder_builder
from second.pytorch.builder import second_builder_for_official_onnx_and_cuda
import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import torchplus


def model_2_onnx(config_path,
                 model_dir,
                 ckpt_path=None):

    model_dir = pathlib.Path(model_dir)

    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    model_cfg = config.model.second
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)

    net = second_builder_for_official_onnx_and_cuda.build(model_cfg, voxel_generator, target_assigner)

    # since the model is changed, dont restore first
    if ckpt_path is None:
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)

    # print(net)
    # convert model to onnx
    dummy_dev_pillar_x_ = torch.randn(1, 1, 12000, 100, device='cuda')
    dummy_dev_pillar_y_ = torch.randn(1, 1, 12000, 100, device='cuda')
    dummy_dev_pillar_z_ = torch.randn(1, 1, 12000, 100, device='cuda')
    dummy_dev_pillar_i_ = torch.randn(1, 1, 12000, 100, device='cuda')
    dummy_dev_num_points_per_pillar_ = torch.randn(1, 1, 12000, 1, device='cuda')
    dummy_dev_x_coors_for_sub_shaped_ = torch.randn(1, 1, 12000, 100, device='cuda')
    dummy_dev_y_coors_for_sub_shaped_ = torch.randn(1, 1, 12000, 100, device='cuda')
    dummy_dev_pillar_feature_mask_ = torch.randn(1, 1, 12000, 100, device='cuda')
    dummy_dev_scattered_feature = torch.randn(1, 64, 496, 432, device='cuda')
    net.cuda()
    net.eval()
    torch.onnx.export(net.voxel_feature_extractor,
                      (dummy_dev_pillar_x_,
                       dummy_dev_pillar_y_,
                       dummy_dev_pillar_z_,
                       dummy_dev_pillar_i_,
                       dummy_dev_num_points_per_pillar_,
                       dummy_dev_x_coors_for_sub_shaped_,
                       dummy_dev_y_coors_for_sub_shaped_,
                       dummy_dev_pillar_feature_mask_
                       ),
                       "./pfe_test.onnx",
                       verbose=False)
    torch.onnx.export(net.rpn,
                      dummy_dev_scattered_feature,
                      "./rpn_test.onnx",
                      verbose=False)


def test_onnx_for_trt(onnx_path,
                      config_path,
                      model_dir,
                      ckpt_path=None):
    dummy_dev_pillar_x_ = np.random.random(size=(1, 1, 12000, 100)).astype(np.float32)
    dummy_dev_pillar_y_ = np.random.random(size=(1, 1, 12000, 100)).astype(np.float32)
    dummy_dev_pillar_z_ = np.random.random(size=(1, 1, 12000, 100)).astype(np.float32)
    dummy_dev_pillar_i_ = np.random.random(size=(1, 1, 12000, 100)).astype(np.float32)
    dummy_dev_num_points_per_pillar_ = np.random.random(size=(1, 1, 12000, 1)).astype(np.float32)
    dummy_dev_x_coors_for_sub_shaped_ = np.random.random(size=(1, 1, 12000, 100)).astype(np.float32)
    dummy_dev_y_coors_for_sub_shaped_ = np.random.random(size=(1, 1, 12000, 100)).astype(np.float32)
    dummy_dev_pillar_feature_mask_ = np.random.random(size=(1, 1, 12000, 100)).astype(np.float32)

    model = onnx.load(onnx_path)
    engine = backend.prepare(model, device='CUDA:0', max_batch_size=1)
    print("model read success")
    print()
    output_data = engine.run((dummy_dev_pillar_x_,
                              dummy_dev_pillar_y_,
                              dummy_dev_pillar_z_,
                              dummy_dev_pillar_i_,
                              dummy_dev_num_points_per_pillar_,
                              dummy_dev_x_coors_for_sub_shaped_,
                              dummy_dev_y_coors_for_sub_shaped_,
                              dummy_dev_pillar_feature_mask_
                              ))

    # ##########compare with pytorch output #########################
    for i in range(len(output_data)):
        print(output_data[i].shape)
    print(output_data[0][0, 0, 0:100])

    model_dir = pathlib.Path(model_dir)
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    model_cfg = config.model.second
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    net = second_builder_for_official_onnx_and_cuda.build(model_cfg, voxel_generator, target_assigner)
    net.cuda()
    net.eval()

    # since the model is changed, dont restore first
    if ckpt_path is None:
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)

    dummy_dev_pillar_x_ = torch.as_tensor(dummy_dev_pillar_x_, device="cuda")
    dummy_dev_pillar_y_ = torch.as_tensor(dummy_dev_pillar_y_, device="cuda")
    dummy_dev_pillar_z_ = torch.as_tensor(dummy_dev_pillar_z_, device="cuda")
    dummy_dev_pillar_i_ = torch.as_tensor(dummy_dev_pillar_i_, device="cuda")
    dummy_dev_num_points_per_pillar_ = torch.as_tensor(dummy_dev_num_points_per_pillar_, device="cuda")
    dummy_dev_x_coors_for_sub_shaped_ = torch.as_tensor(dummy_dev_x_coors_for_sub_shaped_, device="cuda")
    dummy_dev_y_coors_for_sub_shaped_ = torch.as_tensor(dummy_dev_y_coors_for_sub_shaped_, device="cuda")
    dummy_dev_pillar_feature_mask_ = torch.as_tensor(dummy_dev_pillar_feature_mask_, device="cuda")
    output_pytorch = net.voxel_feature_extractor(dummy_dev_pillar_x_, dummy_dev_pillar_y_, dummy_dev_pillar_z_,
                                                 dummy_dev_pillar_i_, dummy_dev_num_points_per_pillar_,
                                                 dummy_dev_x_coors_for_sub_shaped_, dummy_dev_y_coors_for_sub_shaped_,
                                                 dummy_dev_pillar_feature_mask_)

    print(output_pytorch[0, 0, 0:100])




onnx_path = "./pfe_test.onnx"
model_2_onnx("../configs/pointpillars/car/xyres_16.proto", "../../kitti_models")
test_onnx_for_trt(onnx_path, "../configs/pointpillars/car/xyres_16.proto", "../../kitti_models")



"""
model = onnx.load("./pfe_test.onnx")
# Check that the IR is well formed
print(onnx.checker.check_model(model))
# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
model = onnx.load("./pfe_official.onnx")
# Check that the IR is well formed
print(onnx.checker.check_model(model))
# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
"""


"""
model = onnx.load("./pfe_test.onnx")
# Check that the IR is well formed
print(onnx.checker.check_model(model))
# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
"""