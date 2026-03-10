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
    destroy_calib_data_loader,
    torch_load_compat,
    extract_state_dict,
)

from pytorch_quantization import nn as quant_nn


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def resolve_export_paths(args):
    onnx_suffix = "_no_quant" if args.no_quant else "_partial_quant"
    default_stem = os.path.splitext(os.path.basename(args.config_file))[0] + onnx_suffix
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.getcwd()
    output_name = args.output_name or default_stem

    for ext in (".onnx", ".trt", ".engine"):
        if output_name.endswith(ext):
            output_name = output_name[: -len(ext)]
            break

    mkdir(output_dir)
    onnx_name = os.path.join(output_dir, f"{output_name}.onnx")
    raw_onnx_name = os.path.join(output_dir, f"{output_name}_raw.onnx")
    trt_name = os.path.join(output_dir, f"{output_name}_bs{args.batch_size}.trt")
    if args.output_dir or args.output_name:
        calib_name = os.path.join(output_dir, f"{output_name}_calib.pth")
    else:
        ckpt_root, _ = os.path.splitext(args.ckpt)
        calib_name = f"{ckpt_root}_calib.pth"
    return raw_onnx_name, onnx_name, trt_name, calib_name


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


def probe_max_quant_input_elements(model, inference_h, inference_w, device):
    """Find max per-sample input elements among quantizable ops at export size."""

    max_elements = 0
    max_module = "N/A"
    hooks = []
    candidate_types = (nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d)

    def make_hook(name):
        def _hook(_module, inputs):
            nonlocal max_elements, max_module
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x) or x.numel() == 0:
                return
            if x.dim() > 0 and x.size(0) > 0:
                per_sample_elements = x[0].numel()
            else:
                per_sample_elements = x.numel()
            if per_sample_elements > max_elements:
                max_elements = int(per_sample_elements)
                max_module = name

        return _hook

    for name, module in model.named_modules():
        if isinstance(module, candidate_types):
            hooks.append(module.register_forward_pre_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, inference_h, inference_w, device=device)
            _ = model(dummy_input)
    finally:
        for hook in hooks:
            hook.remove()
        if was_training:
            model.train()

    return max_elements, max_module


def compute_safe_calib_batches(max_elements_per_sample, batch_size, safety_ratio):
    int32_max = 2_147_483_647
    denom = int(max_elements_per_sample) * int(batch_size)
    if denom <= 0:
        return 1
    safe_batches = int((int32_max * safety_ratio) // denom)
    return max(safe_batches, 1)


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
        "--output_dir",
        type=str,
        default=None,
        help="directory to write exported ONNX/TRT files; defaults to the current working directory",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="base filename for exported ONNX/TRT files without extension",
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--calib_weights", type=str, default=None, help="calib weights")
    parser.add_argument(
        "--calib_batches",
        type=int,
        default=1000,
        help="maximum calibration batches to consume",
    )
    parser.add_argument(
        "--calib_method",
        type=str,
        choices=["entropy", "percentile", "max"],
        default="entropy",
        help="primary calibration method",
    )
    parser.add_argument(
        "--calib_fallback_method",
        type=str,
        choices=["percentile", "max", "none"],
        default="percentile",
        help="fallback method if primary amax calibration fails",
    )
    parser.add_argument(
        "--calib_percentile",
        type=float,
        default=99.99,
        help="percentile value used by percentile calibration",
    )
    parser.add_argument(
        "--calib_safety_ratio",
        type=float,
        default=0.9,
        help="safety ratio for auto-capping calibration batches to avoid int32 overflow",
    )
    parser.add_argument(
        "--calib_input_range",
        type=str,
        choices=["0_1", "0_255"],
        default="0_1",
        help="calibration input scaling: 0_1 divides calibration images by 255 to match deployment preprocessing",
    )
    parser.add_argument(
        "--no_calib_auto_cap",
        action="store_true",
        help="disable auto-capping calibration batches based on inferred tensor size",
    )
    parser.add_argument(
        "--no_quant",
        action="store_true",
        help="skip PTQ entirely and export a float ONNX through this script",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="quant model type(tiny, small, medium)",
    )
    parser.add_argument(
        "--keep_head_fp",
        action="store_true",
        help="keep the final detection head convolutions in full precision",
    )
    parser.add_argument(
        "--quant_op_indices",
        type=int,
        nargs="+",
        default=None,
        help="explicit quantizable op indices to quantize; overrides --model_type presets",
    )
    parser.add_argument(
        "--quantize_all",
        action="store_true",
        help="quantize all Conv/ConvTranspose/MaxPool candidate ops; overrides --model_type presets",
    )
    parser.add_argument(
        "--skip_layers",
        type=str,
        default=None,
        help="comma-separated substrings of quantizable op names to keep in floating point",
    )
    parser.add_argument(
        "--list_quant_ops",
        action="store_true",
        help="print quantizable op indices/names and exit",
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
    parser.add_argument(
        "--no_simplify",
        action="store_true",
        help="skip onnxsim.simplify after raw ONNX export",
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
def trt_export(onnx_path, batch_size, inference_h, inference_w, engine_path=None):
    import tensorrt as trt

    TRT_LOGGER = trt.Logger()
    if engine_path is None:
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
    if not (0 < args.calib_safety_ratio <= 1):
        raise ValueError("--calib_safety_ratio must be in (0, 1].")
    if args.calib_batches <= 0:
        raise ValueError("--calib_batches must be > 0.")
    if args.quantize_all and args.model_type is not None:
        logger.warning(
            "--quantize_all overrides --model_type='{}'. Ignoring model_type.".format(
                args.model_type
            )
        )

    skip_patterns = []
    if args.skip_layers:
        skip_patterns = [p.strip() for p in args.skip_layers.split(",") if p.strip()]
        logger.info("Skipping quantization for layer patterns: {}".format(skip_patterns))

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

    raw_onnx_name, onnx_name, trt_name, calib_name = resolve_export_paths(args)
    logger.info(
        "Export outputs: raw_onnx='{}', onnx='{}', trt='{}', calib='{}'".format(
            raw_onnx_name, onnx_name, trt_name, calib_name
        )
    )

    # Ensure calibration dataloader uses a fixed export resolution (supports rectangular inputs).
    config.test.augment.transform.target_size = (inference_h, inference_w)
    if args.calib_input_range == "0_1":
        config.test.augment.transform.image_mean = [0.0, 0.0, 0.0]
        config.test.augment.transform.image_std = [255.0, 255.0, 255.0]
    else:
        config.test.augment.transform.image_mean = [0.0, 0.0, 0.0]
        config.test.augment.transform.image_std = [1.0, 1.0, 1.0]
    logger.info(
        "Calibration preprocessing: target_size={}x{}, image_mean={}, image_std={}".format(
            inference_h,
            inference_w,
            config.test.augment.transform.image_mean,
            config.test.augment.transform.image_std,
        )
    )

    # build model
    model = build_local_model(config, device)
    ckpt = torch_load_compat(args.ckpt, map_location="cpu")
    model.eval()
    state_dict = extract_state_dict(ckpt)
    load_result = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "loading checkpoint done. missing_keys={}, unexpected_keys={}".format(
            len(load_result.missing_keys), len(load_result.unexpected_keys)
        )
    )
    if load_result.missing_keys:
        logger.warning(
            "Missing checkpoint keys: {}".format(load_result.missing_keys[:20])
        )
    if load_result.unexpected_keys:
        logger.warning(
            "Unexpected checkpoint keys: {}".format(load_result.unexpected_keys[:20])
        )
    model = replace_module(model, nn.SiLU, SiLU)
    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()
    info = get_model_info(model, (inference_h, inference_w))
    logger.info(info)

    # decouple postprocess
    model.head.nms = False

    if args.no_quant:
        logger.warning(
            "Skipping PTQ and exporting float model through partial_quant.py."
        )
        ptq_model = model
    else:
        # 1. do post training quantization
        if args.calib_weights is None:
            calib_data_loader = init_calib_data_loader(config)
            try:
                try:
                    loader_batches = len(calib_data_loader)
                except TypeError:
                    loader_batches = None

                requested_calib_batches = args.calib_batches
                max_elements_per_sample = None
                max_module_name = None
                safe_calib_batches = None
                effective_calib_batches = requested_calib_batches

                if args.no_calib_auto_cap:
                    if loader_batches is not None:
                        effective_calib_batches = min(
                            requested_calib_batches, loader_batches
                        )
                    logger.info(
                        "Calibration auto-cap disabled. requested_batches={}, loader_batches={}, effective_batches={}".format(
                            requested_calib_batches,
                            loader_batches if loader_batches is not None else "unknown",
                            effective_calib_batches,
                        )
                    )
                else:
                    max_elements_per_sample, max_module_name = (
                        probe_max_quant_input_elements(
                            model, inference_h, inference_w, device
                        )
                    )
                    safe_calib_batches = compute_safe_calib_batches(
                        max_elements_per_sample=max_elements_per_sample,
                        batch_size=args.batch_size,
                        safety_ratio=args.calib_safety_ratio,
                    )

                    batch_limits = [requested_calib_batches, safe_calib_batches]
                    if loader_batches is not None:
                        batch_limits.append(loader_batches)
                    effective_calib_batches = min(batch_limits)

                    logger.info(
                        "Calibration batch planning: requested_batches={}, loader_batches={}, safe_batches={}, "
                        "effective_batches={}, max_elements_per_sample={}, max_module='{}'".format(
                            requested_calib_batches,
                            loader_batches if loader_batches is not None else "unknown",
                            safe_calib_batches,
                            effective_calib_batches,
                            max_elements_per_sample,
                            max_module_name,
                        )
                    )
                    if effective_calib_batches < requested_calib_batches:
                        logger.warning(
                            "Calibration batches capped from {} to {}.".format(
                                requested_calib_batches, effective_calib_batches
                            )
                        )

                ptq_model = post_train_quant(
                    model,
                    calib_data_loader,
                    effective_calib_batches,
                    device,
                    calib_method=args.calib_method,
                    fallback_method=args.calib_fallback_method,
                    percentile=args.calib_percentile,
                )
                torch.save(
                    {"model_state_dict": ptq_model.state_dict()},
                    calib_name,
                )
            finally:
                destroy_calib_data_loader()
        else:
            ptq_model = load_quanted_model(
                model, args.calib_weights, device, calib_method=args.calib_method
            )

        # 2. load sensitivity data
        all_ops = list()
        for k, m in ptq_model.named_modules():
            if (
                isinstance(m, quant_nn.QuantConv2d)
                or isinstance(m, quant_nn.QuantConvTranspose2d)
                or isinstance(m, quant_nn.QuantMaxPool2d)
            ):
                all_ops.append((k))

        if args.list_quant_ops:
            for idx, op_name in enumerate(all_ops):
                logger.info(
                    "quant_op[{idx:02d}] = {name}".format(idx=idx, name=op_name)
                )
            return

        if args.quant_op_indices is not None:
            selected_inds = sorted(set(args.quant_op_indices))
            invalid_inds = [
                idx for idx in selected_inds if idx < 0 or idx >= len(all_ops)
            ]
            if invalid_inds:
                raise ValueError(
                    "Invalid --quant_op_indices {}. Valid range is [0, {}].".format(
                        invalid_inds, len(all_ops) - 1
                    )
                )
            if args.keep_head_fp:
                logger.warning(
                    "--keep_head_fp is ignored because --quant_op_indices explicitly selects ops."
                )
            ops_to_quant = [all_ops[idx] for idx in selected_inds]
            logger.info(
                "Using explicit quant op indices: {}".format(
                    ", ".join(str(idx) for idx in selected_inds)
                )
            )
        elif args.quantize_all:
            ops_to_quant = list(all_ops)
            if args.keep_head_fp:
                logger.warning(
                    "Keeping detection head in FP; excluding head ops from full quantization."
                )
                ops_to_quant = [
                    op_name
                    for op_name in ops_to_quant
                    if not (op_name == "head" or op_name.startswith("head."))
                ]
        else:
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
                    "Provide either --no_quant, --quantize_all, --quant_op_indices, or a supported --model_type (tiny, small, medium)."
                )

            all_inds = backbone_inds + neck_inds + head_inds
            if args.keep_head_fp:
                logger.warning(
                    "Keeping detection head in FP; excluding head ops from quantization."
                )
                all_inds = backbone_inds + neck_inds

            ops_to_quant = [all_ops[x] for x in all_inds]

        if skip_patterns:
            before_count = len(ops_to_quant)
            ops_to_quant = [
                op_name
                for op_name in ops_to_quant
                if not any(pattern in op_name for pattern in skip_patterns)
            ]
            logger.info(
                "Applied --skip_layers filter: kept {} / {} selected ops.".format(
                    len(ops_to_quant), before_count
                )
            )
        logger.info(
            "Partial quantization will quantize {} / {} candidate ops.".format(
                len(ops_to_quant), len(all_ops)
            )
        )
        for op_name in ops_to_quant:
            logger.info("quantizing op: {}".format(op_name))

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
            raw_onnx_name,
            **export_kwargs,
        )
    else:
        export_kwargs["dynamo"] = False
        try:
            onnx_export(
                ptq_model,
                dummy_input,
                raw_onnx_name,
                **export_kwargs,
            )
        except TypeError:
            export_kwargs.pop("dynamo", None)
            onnx_export(
                ptq_model,
                dummy_input,
                raw_onnx_name,
                **export_kwargs,
            )
    logger.info("generated raw onnx model named {}".format(raw_onnx_name))

    onnx_model = onnx.load(raw_onnx_name)  # Fix output shape
    if args.no_simplify:
        logger.info("Skipping ONNX simplification because --no_simplify was set.")
    else:
        try:
            import onnxsim

            logger.info("Starting to simplify ONNX...")
            # check_n=0 avoids onnxruntime dependency in minimal environments.
            onnx_model, check = onnxsim.simplify(onnx_model, check_n=0)
            assert check, "check failed"
        except Exception as e:
            logger.info(f"simplify skipped: {e}")
    onnx.save(onnx_model, onnx_name)
    logger.info("generated simplified onnx model named {}".format(onnx_name))

    # 5. export trt
    if args.trt:
        trt_name = trt_export(
            onnx_name,
            args.batch_size,
            inference_h,
            inference_w,
            engine_path=trt_name,
        )
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
