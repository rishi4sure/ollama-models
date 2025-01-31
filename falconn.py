from pdf2image import convert_from_path
import os
import PIL.Image
import requests
from paddleocr import PaddleOCR
import pandas as pd
import json

MODEL_API_URL = "http://192.168.77.35:11434/api/generate"

pdf_path = "test/Block 3-ASI-03_Architectural.pdf"
output_dir = "OCR/llama_models/falcon2"
os.makedirs(output_dir, exist_ok=True)

csv_filepath = "sheet.csv"
sheet_data = pd.read_csv(csv_filepath)
sheet_names = sheet_data['sheet_names'].tolist()
#print(sheet_names)

drawing_no_path = "drawing.csv"
drawing_no_data = pd.read_csv(drawing_no_path)
drawing_no = drawing_no_data['drawing_no'].tolist()
#print(drawing_no)

PIL.Image.MAX_IMAGE_PIXELS = 93312000000
images = convert_from_path(pdf_path, dpi=300)
ocr = PaddleOCR(use_angle_cls=False, lang='en')
extracted_data = []  

for i, image in enumerate(images):
    width, height = image.size

    # Crop image vertically (bottom part)
    crop_area_vertical = (0, 3 * height // 4, width, height)
    cropped_image = image.crop(crop_area_vertical)

    # Crop image horizontally (right part)
    cropped_width, cropped_height = cropped_image.size
    crop_area_horizontal = (cropped_width * 3 // 4, 0, cropped_width, cropped_height)
    cropped_image = cropped_image.crop(crop_area_horizontal)

    cropped_path = os.path.join(output_dir, f"cropped_page_{i + 1}.png")
    cropped_image.save(cropped_path)

    # Perform OCR using PaddleOCR
    result = ocr.ocr(cropped_path, cls=True)

    text = ""
    for line in result[0]:
        text += line[1][0] + "\n"

    text_path = os.path.join(output_dir, f"text_{i + 1}.txt")
    with open(text_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text)

    payload = {
    "model": "deepseek-r1:14b",
    "prompt": ("""  Objective : You are tasked with analyzing the provided structural steel drawing text and extracting specific, structured data. Ensure that your extraction follows the exact instructions below, without making assumptions. Only return the required data, and if any information is not available, provide "NOT MENTIONED" for that specific field.

    Fields to Extract:
        
        1 )  Sheet Number / Drawing Number / Line Number :  The Drawing Number / Sheet Number / Line Number are similar terms.
        Extract this as the Sheet Number/ drawing Number / line number associated with the page. The format will mostly be numeric or alphanumeric. 
        usually it will appear last few lines of the text document. It may be represents in the form of A , for architectural,  S for Structural or WS etc.. 
        take {drawing_no} for your reference.     
        
        2 ) Title of the Page :  Extract the Title of the Page or Project Title as it appears in the text.
        The title will describe the content or context of the drawing and may include terms such as Framing Plan, Structural Details, or Specific Equipment Names.      
        use {sheet_names}  for your reference
        example : AXONOMETRIC,
               STATEMENT OF SPECIAL INSPECTIONS,
               

        """
        
        f"\n\nText:\n{text}\n\nDetails:"
    ),
    "max_tokens": 500,
    "temperature": 0.7,
    }

    data = json.dumps(payload).encode("utf-8")
    response = requests.post(MODEL_API_URL, data=data)
    if response.status_code == 200:
        list_dict_words = []
        for each_word in response.text.split("\n"):
            try:
                data = json.loads(each_word) 
            except:
                pass
            list_dict_words.append(data)
            
    llama_response = " ".join([word['response'] for word in list_dict_words if type(word) == type({})])
    #print(llama_response)
    extracted_info = llama_response.strip()
    info_lines = extracted_info.split("\n")
    page_details = {}

    for line in info_lines:
        if ":" in line:
            key, value = line.split(":", 1)
            page_details[key.strip()] = value.strip()

    extracted_data.append(page_details)     
    print(f"page details is {page_details}")
   
df = pd.DataFrame(extracted_data)
excel_path = os.path.join(output_dir, "extracted_data.xlsx")
df.to_excel(excel_path, index=False)
print(f"Data saved to {excel_path}")