import os
import json
import csv
from pypdf import PdfReader

def load_document(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    text = ""

    if ext in [".txt", ".md", ".py"]:
        text = load_txt(filepath)
    elif ext == ".json":
        text = load_json(filepath)
    elif ext == ".csv":
        text = load_csv(filepath)
    elif ext == ".pdf":
        text = load_pdf(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return text, ext

def load_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()
    
def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)

    return json.dumps(data, indent=2)

def load_csv(filepath):
    rows = []

    with open(filepath, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            rows.append(" | ".join(row))

    return "\n".join(rows)

def load_pdf(filepath):
    reader = PdfReader(filepath)

    text = []

    for page in reader.pages:
        text.append(page.extract_text())

    return "\n".join(text)