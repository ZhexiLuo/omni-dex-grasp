## 🖥️ Server Environments (gsam, hamer)

参考官方文档配置环境，然后安装 server 依赖：
```bash
pip install fastapi uvicorn pydantic hydra-core omegaconf
cd thirdparty/{hamer,Grounded-SAM-2} && pip install -e .
```

## 🔄 Pipeline Environment (omnidexgrasp)

```bash
conda create -n omnidexgrasp python=3.10
conda activate omnidexgrasp
pip install opencv-python requests Pillow hydra-core omegaconf pyyaml numpy
```

## 🚀 Usage

```bash
cd omnidexgrasp

# 🖥️ Start servers (separate terminals)
conda activate gsam && python -m recons.server.gsam      # :6001
conda activate hamer && python -m recons.server.hamer    # :6002

# 🔄 Run pipeline
conda activate omnidexgrasp && python -m recons.run
```

## ❓ TODO
- megapose/da3+foundationpose
