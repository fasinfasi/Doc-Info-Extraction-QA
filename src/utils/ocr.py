import pytesseract
from PIL import Image
from src.config import TESSERACT_PATH 

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""