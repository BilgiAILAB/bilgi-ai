from io import StringIO, BytesIO

from django.core.files.uploadedfile import InMemoryUploadedFile

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage


def pdf_to_text(file):
    def convert(file_):
        pagenums = set()

        output = StringIO()
        manager = PDFResourceManager()
        converter = TextConverter(manager, output, laparams=LAParams())
        interpreter = PDFPageInterpreter(manager, converter)

        infile = file_.open()

        for page in PDFPage.get_pages(infile, pagenums):
            interpreter.process_page(page)
        converter.close()
        text = output.getvalue()
        output.close()
        return text

    def change_file_name_to_txt(filename):
        return ''.join(filename.split(".")[:-1]) + '.txt'

    buf = BytesIO()
    buf.write(convert(file).encode('utf-8'))
    file = InMemoryUploadedFile(buf, 'text/plain', change_file_name_to_txt(file.name), 'text/plain', buf.tell(), 'utf-8')
    return file
