from io import StringIO

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
        infile.close()
        converter.close()
        text = output.getvalue()
        output.close()
        return text

    def change_file_name_to_txt(filename):
        return ''.join(filename.split(".")[:-1]) + '.txt'

    buf = StringIO()
    buf.write(convert(file))
    file = InMemoryUploadedFile(buf, "txt", change_file_name_to_txt(file.name), None, buf.tell(), None)
    return file
