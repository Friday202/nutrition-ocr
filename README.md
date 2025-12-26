# Installation

Python 3.10.19 is used if using virtual env.  

```
conda create -n name python=3.10 -y
conda activate name
pip install transformers==4.29.2 datasets==4.1.1 tokenizers==0.13.3
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

or 
```
conda env create -f donut_env.yml
```


Dataset must be present next to root folder, under name data_ocr and inside "nutris" and "demo".