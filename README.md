# Installation

Python 3.10.19 is used if using virtual env.  

## Virtual enviroment 
```
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Conda
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

# Data
Dataset must be present next to root folder, under name data_ocr and inside "nutris" and "sroie". Further each must contain "img" and "key" folder. Img folder has all the images and key folder has all .jsons for "sroie" and nutris.xslx for "nutris".
