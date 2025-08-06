# Environment Setup

This program required CUDA to run the model.

1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Open Conda Prompt (on Windows)
3. Create and activate virtual environment

```bash
conda create -n venv python=3.8
conda activate venv
```

4. Install PyTorch wtih CUDA 11.8 (compatible with 3060)

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

5. Install other dependencies

```bash
pip install -r requirements.txt
```

6. Download [pre-trained models](https://anu365-my.sharepoint.com/:f:/g/personal/u7764210_anu_edu_au/EjvAvE8D1g9Cq9A8k9pjGVsBCvCk-fAR5pHjZF3weIByyg?e=8UBxuY) and store them under the root directory of this program
7. Run `predict.py` to segment the root image

```bash
python predict.py
```

---

# Original Readme

# improved_unet

log download：链接：https://pan.baidu.com/s/1aF3_gPnu6ELvrr2rIosVWQ?pwd=68gt
提取码：68gt
解压码：improved_unet

# Reference

https://github.com/bubbliiiing/unet-pytorch
