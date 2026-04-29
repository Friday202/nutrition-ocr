from paddleocr import PaddleOCR
import common.helpers as helpers

def run_prediction(ocr, image_path, debug=False):          
    result = ocr.predict(input=image_path)

    # Visualize the results and save the JSON results
    for res in result:
        if debug:
            res.save_to_img("output")
            # res.save_to_json("output")
    return result

def load_model():
    ocr = PaddleOCR(use_textline_orientation=True, lang='sl', device='cpu')
    #ocr = PaddleOCR(
      #  use_doc_orientation_classify=False,
     #   use_doc_unwarping=False,
     #   use_textline_orientation=False,
    #    device='cpu')
    return ocr


if __name__ == "__main__":
    ocr = load_model()    

    base_path = helpers.get_img_folder_path("nutris")
    df = helpers.get_nutris_train_dataframe()
    
    for filename in df["FileName"]:
        image_path = base_path / filename
        image_path = str(image_path)
        print(f"Running prediction on {image_path}...")
        result = run_prediction(ocr, image_path, debug=True)
        print(f"Prediction result for {filename}: {result}")
        break
    