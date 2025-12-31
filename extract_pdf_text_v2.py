
from pypdf import PdfReader
import os
import sys

# Set encoding to utf-8 for output
sys.stdout.reconfigure(encoding='utf-8')

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        return str(e)

pdf_path = 'JBHI_Paper_v1 (3).pdf'
if os.path.exists(pdf_path):
    print(extract_text_from_pdf(pdf_path))
else:
    print(f"File not found: {pdf_path}")
