# 🔧 OmniDexGrasp Environment Installation

## 📦 Create Conda Environment

```bash
# 🚀 Create environment with Python 3.10
conda create -n omnidex python=3.10 -y
conda activate omnidex

# 📦 Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 🔌 Install Thirdparty Modules

```bash
cd /home/zhexi/project/omni/omni-2-4

# 1️⃣ Grounded-SAM-2
cd omnidexgrasp/thirdparty/Grounded-SAM-2
pip install -e .
pip install -e grounding_dino
pip install supervision

# 2️⃣ Depth-Anything-3
cd ../Depth-Anything-3
pip install xformers
pip install -e .
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70

# 3️⃣ HaMeR (Hand Mesh Recovery)
cd ../hamer
pip install -e .[all]

# 4️⃣ FoundationPose (⚠️ 需要 Python 3.9, 建议使用Docker)
# 如需本地安装，参考: omnidexgrasp/thirdparty/FoundationPose/readme.md
```

## 📥 Checkpoints Download

### ✅ 已复制 (Local)
```
checkpoints/
├── grounded_sam/     # 2.1GB ✅
│   ├── sam2.1_hiera_base_plus.pt
│   └── grounding-dino-base/
└── hamer/            # 9.1GB ✅
    ├── hamer_ckpts/
    ├── vitpose_ckpts/
    ├── detectron2/
    └── data/mano/MANO_RIGHT.pkl
```

### 📥 需要下载

#### Depth-Anything-3 (自动下载)
```python
# 模型会通过 HuggingFace 自动下载
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
```

#### FoundationPose (手动下载)
从 Google Drive 下载并放置：
```bash
# 📥 权重文件 -> checkpoints/foundation_pose/weights/
# https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i
# - 2023-10-28-18-33-37 (refiner)
# - 2024-01-11-20-02-45 (scorer)

# 📥 Demo数据 -> checkpoints/foundation_pose/demo_data/
# https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP
```

## ✅ Verify Installation

```bash
conda activate omnidex

# Test imports
python -c "from sam2.build_sam import build_sam2; print('✅ SAM2 OK')"
python -c "from groundingdino.util.inference import Model; print('✅ GroundingDINO OK')"
python -c "from depth_anything_3.api import DepthAnything3; print('✅ Depth-Anything-3 OK')"
python -c "from hamer.models import HAMER; print('✅ HaMeR OK')"
```

## 📁 Project Structure

```
omnidexgrasp/thirdparty/
├── Grounded-SAM-2/      # [submodule] 视频分割跟踪
├── Depth-Anything-3/    # [submodule] 深度估计
├── hamer/               # [submodule] 手部姿态估计
└── FoundationPose/      # [submodule] 物体6D姿态估计

checkpoints/             # 模型权重集中存放
assests/                 # 资源文件
```

## ⚠️ Notes

- **CUDA**: 需要 NVIDIA 驱动支持 CUDA 12.1+
- **FoundationPose**: 推荐使用 Docker (`wenbowen123/foundationpose`)
- **HuggingFace Mirror**: 如遇网络问题，设置 `export HF_ENDPOINT=https://hf-mirror.com`
