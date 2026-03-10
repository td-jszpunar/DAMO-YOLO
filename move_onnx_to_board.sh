#!/bin/bash

# Configuration
# SOURCE_PATH="damoyolo_quant/DAMO-YOLO/damoyolo_tinynasL25_S_da_compat_partial_quant.onnx"
SOURCE_PATH="damoyolo_quant/quant_artifacts/full_quant_v2/damoyolo_tinynasL25_S_da_compat_full_quant_candidate_keep_head_fp.onnx"
REMOTE_USER="<remote_user>" # fill the user
REMOTE_HOST="<remote_host_ip>" # fill the ip
REMOTE_PATH="<remote_destination_path>" # fill the path

# Move the file/directory from current system to other board through scp over ssh
if [ -d "$SOURCE_PATH" ]; then
    # Use -r flag for directories
    scp -r "$SOURCE_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
    echo "Directory transferred successfully to ${REMOTE_HOST}:${REMOTE_PATH}"
elif [ -f "$SOURCE_PATH" ]; then
    scp "$SOURCE_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
    echo "File transferred successfully to ${REMOTE_HOST}:${REMOTE_PATH}"
else
    echo "Error: ${SOURCE_PATH} does not exist or is not accessible"
    exit 1
fi
