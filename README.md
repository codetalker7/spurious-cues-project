# Installation and project setup

Just set up a virtual environment using `conda` or `mamba` (or your
favourite VE manager). We use use CUDA 12.8 and the following version of `torch`:

    # for notchpeak-shared-short (debugging partition)
    python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126

    # for granite
    python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

    #for phi-3 mini model use versions - transformers==4.40.2 accelerate==0.30.1 since transformers 5.3.0 might be too new.
    

Other requirements are mentioned in `requirements.txt`; just do `pip install -r requirements.txt`.

