#!/bin/bash
# run as ec2-user, keep Conda
sudo -u ec2-user bash <<'EOF'
source /opt/conda/bin/activate pytorch_p310     # ≤PyTorch-2.5 still ships with Conda
pip install -U pip

# core deps – tweak CUDA wheel if you change instance family/CUDA version
pip install --no-cache-dir \
    "torch==2.2.*+cu121" --index-url=https://download.pytorch.org/whl/cu121 \
    transformers==4.42.0 accelerate==0.30.0 diffusers==0.26.3 \
    bitsandbytes==0.43.1 einops opencv-python imageio[ffmpeg]

# quick smoke test (will pull 14 B-param model; needs ~70 GB GPU RAM)
python - <<PY
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B",
    torch_dtype="float16",
    trust_remote_code=True
).to("cuda")
print("Pipeline loaded OK")
PY
EOF