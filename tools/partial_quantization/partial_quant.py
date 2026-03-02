#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import os
import argparse
import sys
import re
import traceback

import onnx
import torch
from loguru import logger
from torch import nn

from damo.base_models.core.ops import RepConv, SiLU
from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.utils.model_utils import get_model_info, replace_module
from tools.trt_eval import trt_inference

from tools.partial_quantization.utils import (
    post_train_quant,
    load_quanted_model,
    execute_partial_quant,
    init_calib_data_loader,
    torch_load_compat,
)

from pytorch_quantization import nn as quant_nn


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_inference_size(args):
    """Parse inference size from CLI.

    Supported:
      - --img_size 640
      - --img_size 1080x1920
      - --img_size 1080,1920
      - --img_h 1080 --img_w 1920
    """
    if (args.img_h is None) ^ (args.img_w is None):
        raise ValueError("Both --img_h and --img_w must be provided together.")

    if args.img_h is not None and args.img_w is not None:
        inference_h, inference_w = int(args.img_h), int(args.img_w)
    else:
        size_arg = str(args.img_size).lower().strip()
        if re.fullmatch(r"\d+", size_arg):
            side = int(size_arg)
            inference_h, inference_w = side, side
        else:
            tokens = re.split(r"[x,]", size_arg)
            if len(tokens) != 2 or not all(t.strip().isdigit() for t in tokens):
                raise ValueError(
                    f"Invalid --img_size '{args.img_size}'. Use INT or HxW (e.g. 1080x1920)."
                )
            inference_h, inference_w = int(tokens[0].strip()), int(tokens[1].strip())

    if inference_h <= 0 or inference_w <= 0:
        raise ValueError(
            f"Inference size must be positive, got ({inference_h}, {inference_w})."
        )
    return inference_h, inference_w


def make_parser():
    parser = argparse.ArgumentParser("damo converter deployment toolbox")
    # mode part
    parser.add_argument(
        "--mode", default="onnx", type=str, help="onnx, trt_16 or trt_32"
    )
    # model part
    parser.add_argument(
        "-f",
        "--config_file",
        default=None,
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "--trt", action="store_true", help="whether convert onnx into tensorrt"
    )
    parser.add_argument(
        "--trt_type", type=str, default="fp32", help="one type of int8, fp16, fp32"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="inference image batch nums"
    )
    parser.add_argument(
        "--img_size",
        type=str,
        default="640",
        help="inference image shape (INT for square or HxW, e.g. 1080x1920)",
    )
    parser.add_argument(
        "--img_h",
        type=int,
        default=None,
        help="inference image height (overrides --img_size when used with --img_w)",
    )
    parser.add_argument(
        "--img_w",
        type=int,
        default=None,
        help="inference image width (overrides --img_size when used with --img_h)",
    )
    # onnx part
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output",
        default="scores",
        type=str,
        help="scores output node name of onnx model",
    )
    parser.add_argument(
        "--output_bboxes",
        default="bboxes",
        type=str,
        help="bboxes output node name of onnx model",
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--calib_weights", type=str, default=None, help="calib weights")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="quant model type(tiny, small, medium)",
    )
    parser.add_argument(
        "--sensitivity_file", type=str, default=None, help="sensitivity file"
    )
    parser.add_argument("--end2end", action="store_true", help="export end2end onnx")
    parser.add_argument(
        "--ort", action="store_true", help="export onnx for onnxruntime"
    )
    parser.add_argument(
        "--dynamic_batch",
        action="store_true",
        help="export dynamic batch axis for ONNX input/outputs",
    )
    parser.add_argument("--trt_eval", action="store_true", help="trt evaluation")
    parser.add_argument(
        "--iou-thres", type=float, default=0.65, help="iou threshold for NMS"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.05, help="conf threshold for NMS"
    )
    parser.add_argument(
        "--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


@logger.catch
def trt_export(onnx_path, batch_size, inference_h, inference_w):
    import tensorrt as trt

    TRT_LOGGER = trt.Logger()
    engine_path = onnx_path.replace(".onnx", f"_bs{batch_size}.trt")

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with (
        trt.Builder(TRT_LOGGER) as builder,
        builder.create_network(EXPLICIT_BATCH) as network,
        trt.OnnxParser(network, TRT_LOGGER) as parser,
    ):
        logger.info("Loading ONNX file from path {}...".format(onnx_path))
        with open(onnx_path, "rb") as model:
            logger.info("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                logger.info("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    logger.info(parser.get_error(error))

        builder.max_batch_size = batch_size
        logger.info("Building an engine.  This would take a while...")
        config = builder.create_builder_config()
        config.max_workspace_size = 2 << 30

        config.flags |= 1 << int(trt.BuilderFlag.INT8)
        config.flags |= 1 << int(trt.BuilderFlag.FP16)

        engine = builder.build_engine(network, config)
        try:
            assert engine
        except AssertionError:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            _, line, _, text = tb_info[-1]
            raise AssertionError(
                "Parsing failed on line {} in statement {}".format(line, text)
            )

        logger.info("generated trt engine named {}".format(engine_path))
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        return engine_path


@logger.catch
def main():
    args = make_parser().parse_args()

    logger.info("args value: {}".format(args))
    inference_h, inference_w = parse_inference_size(args)
    logger.info(f"Using inference size: {inference_h}x{inference_w}")

    onnx_name = args.config_file.split("/")[-1].replace(".py", "_partial_quant.onnx")
    # Check device
    cuda = args.device != "cpu" and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.device}" if cuda else "cpu")

    # init config
    config = parse_config(args.config_file)
    config.merge(args.opts)
    if args.batch_size is not None:
        config.test.batch_size = args.batch_size
    else:
        args.batch_size = config.test.batch_size

    # Ensure calibration dataloader uses a fixed export resolution (supports rectangular inputs).
    config.test.augment.transform.target_size = (inference_h, inference_w)

    # build model
    model = build_local_model(config, device)
    ckpt = torch_load_compat(args.ckpt, map_location="cpu")
    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=False)
    logger.info("loading checkpoint done.")
    model = replace_module(model, nn.SiLU, SiLU)
    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()
    info = get_model_info(model, (inference_h, inference_w))
    logger.info(info)

    # decouple postprocess
    model.head.nms = False

    # 1. do post training quantization
    if args.calib_weights is None:
        calib_data_loader = init_calib_data_loader(config)
        ptq_model = post_train_quant(model, calib_data_loader, 1000, device)
        torch.save(
            {"model_state_dict": ptq_model.state_dict()},
            args.ckpt.replace(".pth", "_calib.pth"),
        )
    else:
        ptq_model = load_quanted_model(model, args.calib_weights, device)

    # 2. load sensitivity data
    all_ops = list()
    for k, m in ptq_model.named_modules():
        if (
            isinstance(m, quant_nn.QuantConv2d)
            or isinstance(m, quant_nn.QuantConvTranspose2d)
            or isinstance(m, quant_nn.MaxPool2d)
        ):
            all_ops.append((k))

    quant_model = args.model_type
    if quant_model == "tiny":
        backbone_inds = list(range(24))
        neck_inds = []
        head_inds = list(range(74, 80))
    elif quant_model == "small":
        backbone_inds = list(range(30))
        neck_inds = (
            list(range(30, 31))
            + list(range(32, 40))
            + list(range(40, 41))
            + list(range(42, 49))
            + list(range(50, 51))
            + list(range(52, 59))
            + list(range(60, 61))
            + list(range(62, 69))
            + list(range(70, 71))
            + list(range(72, 79))
        )
        head_inds = list(range(80, 86))
    elif quant_model == "medium":
        backbone_inds = (
            list(range(5))
            + list(range(6, 15))
            + list(range(16, 33))
            + list(range(34, 46))
            + list(range(47, 48))
        )
        neck_inds = []
        head_inds = list(range(108, 114))
    else:
        raise ValueError(
            "unsupported model type in requested schema(tiny, small, medium)"
        )

    all_inds = backbone_inds + neck_inds + head_inds

    quantable_sensitivity = [all_ops[x] for x in all_inds]
    ops_to_quant = [qops for qops in quantable_sensitivity]

    # 3. only quantize ops in quantable_ops list
    execute_partial_quant(ptq_model, ops_to_quant=ops_to_quant)

    # 4. ONNX export
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    dummy_input = torch.randn(args.batch_size, 3, inference_h, inference_w).to(device)
    _ = ptq_model(dummy_input)
    onnx_export = getattr(torch.onnx, "export", None)
    if onnx_export is None:
        onnx_export = torch.onnx._export
    legacy_onnx_export = getattr(getattr(torch.onnx, "utils", None), "export", None)

    export_kwargs = dict(
        verbose=False,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=[args.input],
        output_names=[args.output, args.output_bboxes],
        opset_version=13,
    )
    if args.dynamic_batch:
        export_kwargs["dynamic_axes"] = {
            args.input: {0: "batch_size"},
            args.output: {0: "batch_size"},
            args.output_bboxes: {0: "batch_size"},
        }
    # Prefer legacy exporter path to avoid torch.export graph capture issues with fake quant.
    if legacy_onnx_export is not None:
        legacy_onnx_export(
            ptq_model,
            dummy_input,
            onnx_name,
            **export_kwargs,
        )
    else:
        export_kwargs["dynamo"] = False
        try:
            onnx_export(
                ptq_model,
                dummy_input,
                onnx_name,
                **export_kwargs,
            )
        except TypeError:
            export_kwargs.pop("dynamo", None)
            onnx_export(
                ptq_model,
                dummy_input,
                onnx_name,
                **export_kwargs,
            )
    onnx_model = onnx.load(onnx_name)  # Fix output shape
    try:
        import onnxsim

        logger.info("Starting to simplify ONNX...")
        # check_n=0 avoids onnxruntime dependency in minimal environments.
        onnx_model, check = onnxsim.simplify(onnx_model, check_n=0)
        assert check, "check failed"
    except Exception as e:
        logger.info(f"simplify skipped: {e}")
    onnx.save(onnx_model, onnx_name)
    logger.info("generated onnx model named {}".format(onnx_name))

    # 5. export trt
    if args.trt:
        trt_name = trt_export(onnx_name, args.batch_size, inference_h, inference_w)
        # 6. trt eval
        if args.trt_eval:
            if inference_h != inference_w:
                raise ValueError(
                    "trt_eval in this tool currently expects square img_size. "
                    "Use export only, or extend tools/trt_eval.py for HxW support."
                )
            logger.info("start trt inference on coco validataion dataset")
            trt_inference(
                config,
                trt_name,
                inference_h,
                args.batch_size,
                args.conf_thres,
                args.iou_thres,
                args.end2end,
            )


if __name__ == "__main__":
    main()
