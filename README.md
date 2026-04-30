## Virtual enviroment Donut
```
py -3.10 -m venv .donut
.venv\Scripts\activate.ps1
python -m pip install -r requirements_donut.txt
python -m pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Virtual enviroment Qwen
```
py -3.11 -m venv .venv_qwen
.venv_qwen\Scripts\activate.ps1
python -m pip install -r requirments_qwen.txt
python -m pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Virtual enviroment PaddleOcr
Instrucions for install are written at: 

# Data
Dataset must be present next to root folder, under name data_ocr and inside "nutris" and "sroie". Further each must contain "img" and "key" folder. Img folder has all the images and key folder has all .jsons for "sroie" and nutris.xslx for "nutris".
