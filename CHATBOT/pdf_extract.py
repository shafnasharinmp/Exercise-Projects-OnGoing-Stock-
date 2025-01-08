import PyPDF2

def extract_text_from_pdfs(pdf_paths):
    text = []
    for path in pdf_paths:
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pdf_text = ''
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                pdf_text += page.extract_text()
            text.append(pdf_text)
    return text
