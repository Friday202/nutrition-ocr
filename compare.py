# the final scrip to coampre vision OCR vs segmentation OCR on 1k common samples
# each method paddleOCR and DOnut create their own .csv files 
import common.helpers as helpers


if __name__ == "__main__":
    data = helpers.get_nutris_test_dataframe()
    print(data.head())
