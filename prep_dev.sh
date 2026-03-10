#!/bin/sh

export PYTHONPATH=$PWD:$PYTHONPATH

# uv run python tools/partial_quantization/partial_quant.py \
#   -f /home/jakub/damoyolo_quant/DAMO-YOLO/configs/damoyolo_tinynasL25_S_da_compat.py \
#   -c /home/jakub/damoyolo_quant/model_orig.pth \
#   --batch_size 1 \
#   --img_size 1080x1920 \
#   --input input \
#   --output scores \
#   --output_bboxes bboxes \
#   --calib_batches 500 \
#   --calib_method entropy \
#   --calib_fallback_method percentile \
#   --calib_percentile 99.99 \
#   --calib_safety_ratio 0.9 \
#   --calib_input_range 0_1 \
#   --model_type small \
#   --dynamic_batch


# The command below doesnt work with:
# --calib_weights /home/jakub/damoyolo_quant/model_calib.pth \

# uv run python tools/partial_quantization/partial_quant.py \
#   -f configs/damoyolo_tinynasL25_S_da_compat.py \
#   -c /home/jakub/damoyolo_quant/model_orig.pth \
#   --batch_size 1 \
#   --img_size 1080x1920 \
#   --input input \
#   --output scores \
#   --output_bboxes bboxes \
#   --quantize_all \
#   --output_dir /home/jakub/damoyolo_quant/quant_artifacts/full_quant \
#   --output_name damoyolo_tinynasL25_S_da_compat_full_quant_candidate


uv run tools/partial_quantization/partial_quant.py \
  -f configs/damoyolo_tinynasL25_S_da_compat.py \
  -c /home/jakub/damoyolo_quant/model_orig.pth \
  --batch_size 1 \
  --img_size 1080x1920 \
  --input input \
  --output scores \
  --output_bboxes bboxes \
  --quantize_all \
  --keep_head_fp \
  --no_simplify \
  --output_dir /home/jakub/damoyolo_quant/quant_artifacts/full_quant_v2 \
  --output_name damoyolo_tinynasL25_S_da_compat_full_quant_candidate_keep_head_fp
