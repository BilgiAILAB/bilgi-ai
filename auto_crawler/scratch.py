# -*- coding: <utf-8> -*-
import requests
import re
import sys
import io
import os
import time
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from bs4 import BeautifulSoup
from random import randint


# api 이용예시


class pdf2txt():
    def __init__(self, search_list, download_dir=None, txt_dir=None, pages=5, start_date=None, end_date=None,
                 filetype='pdf'):
        self.search_list = search_list
        self.pages = pages
        self.start_date = start_date
        self.end_date = end_date
        self.titlelist = []

        if download_dir == None:
            self.download_dir = os.path.join("../project", "download", "")
        else:
            self.download_dir = download_dir

        if txt_dir == None:
            self.txt_dir = os.path.join(self.download_dir, "txt", "")
        else:
            self.txt_dir = txt_dir

        if start_date == None and end_date == None:
            self.enable_date = False
        else:
            self.enable_date = True

        self.filetype = filetype

        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.txt_dir, exist_ok=True)

    # ------------------------ 실제 API 사용단 --------------------------#
    def download_search_data(self, time_sleep=True):
        for search in self.search_list:
            links = self.get_link(search, filetype=self.filetype, start_date=self.start_date, end_date=self.end_date,
                                  pages=self.pages, show_links=True)
            self.download_pdf_from_links(search, links, self.filetype, self.download_dir)
            print("")
            if time_sleep:
                time.sleep(randint(1, 2))

    def convert_pdfs(self, to_json=False):
        for search in self.search_list:
            print("----converting", search, "pdfs to txt----")
            os.makedirs(os.path.join(self.txt_dir, search), exist_ok=True)
            self.convertMultiple(os.path.join(self.download_dir, search, ""), os.path.join(self.txt_dir, search, ""),
                                 to_json=to_json)
            print("")

    def concate_all_txt(self):
        for search in self.search_list:
            self.concatFiles(search, self.txt_dir, os.path.join(self.txt_dir, search, ""))

    def is_downloadable(self, url):
        try:
            h = requests.head(url, allow_redirects=False)
            header = h.headers
            content_type = header.get('content-type')
            if 'text' in content_type.lower():
                return False
            if 'html' in content_type.lower():
                return False
            return True
        except:
            return False
        # ------------------------ 실제 API 사용단 --------------------------#

    # ------------------------ 아래부터는 클래스 내부 이용함수 ----------------------#
    def get_link(self, search_keyword, filetype, start_date, end_date, pages, enable_date=False, show_links=False):
        linklist = []
        titlelist = []
        print("search keyword : ", search_keyword)
        print("expected time(getting links) : ", pages * 4, "seconds~", pages * 5, "seconds")
        print("----getting links----")
        for page in range(0, pages * 10, 10):
            params = {}
            params['as_epq'] = search_keyword
            if enable_date:
                params['tbs'] = "cdr:1,cd_min:" + start_date + ",cd_max:" + end_date
            params['start'] = str(page)
            params['as_filetype'] = filetype
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36"}
            html = requests.get("https://www.google.com/search", params=params, headers=headers)
            print(html.url)
            # print(html.text)
            soup = BeautifulSoup(html.text, 'html.parser')
            # soup = BeautifulSoup(html.text, 'lxml')

            for result_table in soup.findAll("div", {"class": "g"}):
                a_click = result_table.find("a")
                if self.is_downloadable(a_click['href']):
                    if show_links == True:
                        print(str(a_click.renderContents())[2:-1], end='  ')
                        print(a_click['href'])
                    linklist.append(a_click['href'])
                    titlelist.append(str(a_click.renderContents())[2:-1])

            time.sleep(randint(1, 2))

        self.titlelist = titlelist
        return linklist

    def download_pdf_from_links(self, search, links, filetype, directory):
        print("----downloading----")
        print("files to download : ", len(links))
        for num, link in enumerate(links):
            try:
                r = requests.get(link, stream=True)
                os.makedirs(os.path.join(directory, search), exist_ok=True)
                with open(os.path.join(directory, search, "") + str(num + 1) + '.' + filetype, 'wb') as f:
                    f.write(r.content)
                print(str((num + 1) / len(links) * 100) + "% done")
            except:
                print("download error on : ", link)

    # (c) 2016 Masha Gorkovenko stanford.edu
    # converts pdf, returns its text content as a string
    def convert(self, fname, pages=None):
        print(fname)
        if not pages:
            pagenums = set()
        else:
            pagenums = set(pages)

        output = io.StringIO()
        manager = PDFResourceManager()
        converter = TextConverter(manager, output, laparams=LAParams())
        interpreter = PDFPageInterpreter(manager, converter)

        infile = open(fname, 'rb')
        print(infile)
        for page in PDFPage.get_pages(infile, pagenums):
            interpreter.process_page(page)
        infile.close()
        converter.close()
        text = output.getvalue()
        output.close
        print(text)
        return text

    # converts all pdfs in directory pdfDir, saves all resulting txt files to txtdir
    def convertMultiple(self, pdfDir, txtDir, to_json=False):
        if to_json:
            import pandas as pd
            out = pd.DataFrame()
            out['Filename'] = ""
            out['Title'] = ""
            out['Content'] = ""
        print("----converting started----")
        if pdfDir == "": pdfDir = os.getcwd() + "\\"  # if no pdfDir passed in
        for i, pdf in enumerate(os.listdir(pdfDir)):  # iterate through pdfs in pdf directory
            fileExtension = pdf.split(".")[-1]
            if fileExtension == "pdf":
                pdfFilename = pdfDir + pdf
                print("converting :", pdfFilename, end='  ')
                text = self.convert(pdfFilename)  # get string of text content of pdf
                textFilename = txtDir + pdf + ".txt"
                textFile = open(textFilename, "w", encoding='utf-8')  # make text file
                textFile.write(text)  # write text to text file
                if to_json:
                    out = out.append([pd.DataFrame([[pdfFilename, self.titlelist[int(pdf[:-4]) - 1], text]],
                                                   columns=['Filename', 'Title', 'Content'])], ignore_index=True)
                print("done  ", str(round((i + 1) / len(os.listdir(pdfDir)) * 100, 2)) + "%")

        if to_json:
            json = out.to_json()
            with open(txtDir + 'json_out.txt', 'w', encoding='utf-8') as f:
                f.write(json)

    def concatFiles(self, corp, txt_dir, path):
        files = os.listdir(path)
        concat = ''
        for file in files:
            try:
                # print(path+file)
                with open(path + file, encoding="utf-8") as infile:
                    for line in infile:
                        concat += ' '.join(bytes(line, 'utf-8').decode('utf-8', 'ignore').splitlines())
                        # concat += ''.join(open(path + file, encoding="utf-8").read().splitlines())

            except:  # utf-8로 decode가 불가능할 때 뜨는 에러인 듯?
                print("error on : " + file)
                print("On mac, .DS_Store -> system file")

        concat = re.compile('^\w\s').sub('', concat)

        with open(txt_dir + corp + ".txt", "w+", encoding='utf-8') as fo:
            fo.write(concat)
        print(txt_dir + corp + ".txt")


if __name__ == "__main__":
    pdfcrawler = pdf2txt(["pozisyonel vertigo"], pages=10)  # list의 항목을 각각 검색함
    pdfcrawler.download_search_data()
    pdfcrawler.convert_pdfs()
    pdfcrawler.concate_all_txt()
