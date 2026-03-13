# Installation and project setup

Just set up a virtual environment using `conda` or `mamba` (or your
favourite VE manager). We use use CUDA 12.8 and the following version of `torch`:

    python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126

Other requirements are mentioned in `requirements.txt`; just do `pip install -r requirements.txt`.
