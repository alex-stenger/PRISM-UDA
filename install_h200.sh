#!/bin/bash
# ============================================================
# PRISM-UDA Install Script for NVIDIA H200 (sm_90)
# Tested on: H200, CUDA driver 12.x, Python 3.8.5
# ============================================================

set -e  # exit on error

CONDA_BASE=${1:-"$HOME/miniconda3"}  # pass your miniconda path as argument, or edit this
ENV_NAME="prism_h200"
PRISM_PATH=${2:-"$(pwd)"}  # pass your PRISM-UDA path as argument, or edit this

echo "============================================"
echo "Using conda at: $CONDA_BASE"
echo "ENV name: $ENV_NAME"
echo "PRISM-UDA path: $PRISM_PATH"
echo "============================================"

# ------------------------------------------------------------
# 1. Init conda
# ------------------------------------------------------------
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ------------------------------------------------------------
# 2. Create environment
# ------------------------------------------------------------
conda create -n $ENV_NAME python=3.8.5 -y
conda activate $ENV_NAME

# ------------------------------------------------------------
# 3. Install PyTorch 2.1 + CUDA 12.1 (supports sm_90 / H200)
# ------------------------------------------------------------
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------
# 4. Install mmcv-full 1.7.2 built for torch2.1/cu121
# ------------------------------------------------------------
pip install mmcv-full==1.7.2 \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# ------------------------------------------------------------
# 5. Install remaining dependencies
#    (Pillow 8.4.0 instead of 8.3.1 — torchvision blacklists 8.3.*)
# ------------------------------------------------------------
pip install \
  cityscapesscripts==2.2.0 \
  cycler==0.10.0 \
  gdown==4.2.0 \
  humanfriendly==9.2 \
  kiwisolver==1.2.0 \
  kornia==0.5.8 \
  matplotlib==3.4.2 \
  numpy==1.19.2 \
  opencv-python==4.4.0.46 \
  pandas==1.1.3 \
  Pillow==8.4.0 \
  prettytable==2.1.0 \
  pyparsing==2.4.7 \
  pytz==2020.1 \
  PyYAML==5.4.1 \
  scipy==1.6.3 \
  seaborn==0.11.1 \
  timm==0.3.2 \
  tqdm==4.48.2 \
  wcwidth==0.2.5 \
  yapf==0.31.0 \
  --extra-index-url https://download.pytorch.org/whl/cu121

# upgrade typing_extensions so torch 2.1 can import correctly
pip install --upgrade typing_extensions

# ------------------------------------------------------------
# 6. Patch timm 0.3.2 — torch._six.container_abcs removed in torch 1.9+
# ------------------------------------------------------------
TIMM_HELPERS="$CONDA_BASE/envs/$ENV_NAME/lib/python3.8/site-packages/timm/models/layers/helpers.py"
sed -i 's/from torch._six import container_abcs/import collections.abc as container_abcs/' "$TIMM_HELPERS"
echo "✔ timm patch applied"

# ------------------------------------------------------------
# 7. Patch mmcv — passes int instead of torch.device to _get_stream
#    (torch 2.x API change)
# ------------------------------------------------------------
MMCV_FUNCTIONS="$CONDA_BASE/envs/$ENV_NAME/lib/python3.8/site-packages/mmcv/parallel/_functions.py"
sed -i 's/streams = \[_get_stream(device) for device in target_gpus\]/streams = [_get_stream(torch.device("cuda", device) if isinstance(device, int) else device) for device in target_gpus]/' "$MMCV_FUNCTIONS"
echo "✔ mmcv _functions.py patch applied"

# ------------------------------------------------------------
# 8. Patch PRISM-UDA mmcv version check
#    (code requires <=1.4.0 but we need 1.7.2 for torch2.x compat)
# ------------------------------------------------------------
sed -i "s/MMCV_MAX = '1.4.0'/MMCV_MAX = '1.7.2'/" "$PRISM_PATH/mmseg/__init__.py"
echo "✔ PRISM-UDA mmcv version check patched"

# ------------------------------------------------------------
# 9. Verify
# ------------------------------------------------------------
echo ""
echo "============================================"
echo "Verifying installation..."
echo "============================================"
python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
import mmcv
print('mmcv:', mmcv.__version__)
from mmseg.apis import set_random_seed, train_segmentor
print('mmseg: OK')
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
print('timm: OK')
print('')
print('All good! Ready to train on H200.')
"

echo ""
echo "============================================"
echo "Installation complete!"
echo "To activate: source $CONDA_BASE/etc/profile.d/conda.sh && conda activate $ENV_NAME"
echo "============================================"

#if it does not work, also we can
#pip install open_clip_torch==2.24.0 timm==0.9.16
#pip install transformers
