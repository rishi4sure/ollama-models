from pdf2image import convert_from_path
import os
import PIL.Image
import requests
from paddleocr import PaddleOCR
import pandas as pd
import json


MODEL_API_URL = "http://192.168.77.35:11434/api/generate"


pdf_path = "test/multipdf.pdf"
output_dir = "OCR/llama_models/Structural_llama"
os.makedirs(output_dir, exist_ok=True)

PIL.Image.MAX_IMAGE_PIXELS = 93312000000
images = convert_from_path(pdf_path, dpi=300)
ocr = PaddleOCR(use_angle_cls=True, lang='en')
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
    "model": "joreilly86/structural_llama:latest",
    "prompt": ("""  Objective : You are tasked with analyzing the provided structural steel drawing text and extracting specific, structured data. Ensure that your extraction follows the exact instructions below, without making assumptions. Only return the required data, and if any information is not available, provide "NOT MENTIONED" for that specific field.

        Fields to Extract:
            1 )  Job Number : The Job Number may be referred to by multiple terms: "Job Number," "Plan Number," or "Project Number." These are considered synonymous.
            It should be a number or alphanumeric string (e.g., 20024113-ST3028-001, 1916.702, MEI 17017).
            Extract this value and present it exactly as found in the text.
            
            2 )  Drawing Number (or Line Number) :  The Drawing Number (or Line Number) can also be referred to as the Sheet Number or similar terms.
            Extract this as the drawing/line number associated with the page.  If the Drawing Number is missing, use the Job Number as the Drawing Number or vice versa. The format will mostly be numeric or alphanumeric (e.g.,  LP150-SF-0008J251, A0.30, DWG NO 1010).
            
            
            
            3 ) Title of the Page :  Extract the Title of the Page or Project Title as it appears in the text.
            The title will describe the content or context of the drawing and may include terms such as Framing Plan, Structural Details, or Specific Equipment Names.
            Example Titles:
            THERMAL CRUDE TO CHEMICAL COMPLEX
            STRUCTURAL DESIGN - FRAMING PLAN
            HRSG ELECTRICAL ENCLOSURE SUPPORT STEEL
            COLUMN AND BASE PLATE PLAN AND DETAILS
            If no title is present, output NOT MENTIONED.
            
            4 ) Date :  Extract the most recent date from the text, formatted as DD/MM/YYYY.
            If there are multiple dates, focus only on the most recently modified or updated date.
            If no date is found, output NOT MENTIONED.
                
        Extraction Guidelines:
            Date Extraction:

                If there are multiple dates within the text, return the most recent date (latest modified date).
                The format of the date should be DD/MM/YYYY. Only output the date itself, nothing else.
            
            Fallback Rules (if data is missing):

                If you do not find a Job Number, Drawing Number, or Sheet Number, use the Drawing Number/Line Number as both the Job Number and Drawing Number for that page.
                If you do not find a Drawing Number, Line Number, or Job Number, use Job Number/Plan Number/Project No as both the Job Number and Drawing Number.
       
             Special Handling of Missing Information:

                If any of the required fields (Job Number, Drawing Number, Title, or Date) are missing, return NOT MENTIONED for that specific field.
       
        Example:
               
            For a given page, the extracted data should look like the following:

            Job Number: 20024113-ST3028-001
            Drawing Number: S101
            Title of the Page: STRUCTURAL DESIGN - FRAMING PLAN
            Date: 12/04/2023
               
        Additional Notes:
               
            Do not make assumptions about the data. Extract exactly what is given in the raw text.
            Your output must consist of one row of structured data per page. Each page can only have one Job Number, Drawing Number, Title, and Date.
            Use the provided guidelines to handle cases where certain data points are missing."""
        
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
    #print(f"page details is {page_details}")
    

df = pd.DataFrame(extracted_data)
excel_path = os.path.join(output_dir, "extracted_data.xlsx")
df.to_excel(excel_path, index=False)

print(f"Data saved to {excel_path}")

