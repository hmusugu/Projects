# -*- coding: utf-8 -*-
"""
Extracting text from PDF

Created on Fri Feb 21 09:38:01 2020

@author: amasinha
"""
### PART1 - Water Depth Identifying

# importing required modules 
import PyPDF2 
  
# creating a pdf file object 
pdfFileObj = open('MD2.pdf', 'rb') 
  
# creating a pdf reader object 
pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
  
# printing number of pages in pdf file 
print(pdfReader.numPages) 
  
# creating a page object 
pageObj = pdfReader.getPage(0) 
  
## extracting text from page 
#print(pageObj.extractText()) 
  

#discerning the number of pages will allow us to parse through all the pages
num_pages = pdfReader.numPages
count = 0
extracted_text = ""

while count < num_pages:
    pageObj = pdfReader.getPage(count)
    count +=1
    extracted_text += pageObj.extractText()    


import re
define_words = 'water depth'
w_depth2 = re.findall(r"([^.]*?%s[^.]*\.)" % define_words,extracted_text.lower())  

#Making a copy of the extracted text
txt = extracted_text

#Remove . after ft, collect sentences with water depth, remove all chararers from each sentence return numbers before ft.

#Removing the dot after ft
txt = txt.replace('ft.','ft')

#Now, collect all sentences with the word "water depth" in them
define_words = 'water depth'
w_depth2 = re.findall(r"([^.]*?%s[^.]*\.)" % define_words,txt.lower()) 

#Removing everything that is on the left side of the word Water Depth and the word Water Depth itself
w_depth3 = []
for i2 in range(len(w_depth2)):
    splitting = w_depth2[i2].split('water depth')
    w_depth3.append(splitting[1])
    
#Removing commas between the digits    
w_depth4 = []
for i3 in range(len(w_depth3)):
    w_depth4.append(w_depth3[i3].replace(',',''))
    
#Finally printing out the numbers before ft
final = []
for i1 in range(len(w_depth2)):
    final.append(re.findall(r"(\d+) ft", w_depth4[i1]))



 '''   
    
import tabula
from tabula import wrapper


df = tabula.read_pdf("MD2.pdf")

# output just the first table in the PDF to a CSV
tabula.convert_into(file, "iris_first_table.csv")
 
# output all the tables in the PDF to a CSV
file = "C:/Users/hmusugu/Desktop/NLP/MD2.pdf"
tabula.convert_into(file, "tables.csv", pages = 'all')

tabula.convert_into("MD2.pdf", "output.csv", output_format="csv", pages=47)
'''


### PART 2 - Get All tables from the pdf

import camelot


tables = []
for i in range(45,48):
    tables.append(camelot.read_pdf("latest.pdf",pages = '%s'%i))
    

#tables = camelot.read_pdf(file, pages = 'all')

tables = camelot.read_pdf("MD2.pdf",pages = '33', flavor = 'stream')

tables.export("camelot_tables.xlsx", f = "excel")

tables[1].export("camelot_tables.xlsx", f = "excel")

prs.save('{}.pptx'.format(proj))

df = tabula.read_pdf(file, encoding = 'ISO-8859-1', 
         stream=True, area = "81.106,302.475,384.697,552.491", pages = [38] , guess = False,  pandas_options={'header':None})



















#Create Phrase Matcher Object using spacy package
'''    
import spacy

nlp = spacy.load('en_core_web_sm')

from spacy.matcher import PhraseMatcher
phrase_matcher = PhraseMatcher(nlp.vocab)

#Create a list of phrases to match and then convert the list to spaCy NLP documents
phrases = ['water depth','Water Depth','Water Depths']

patterns = [nlp(text) for text in phrases]

#add your phrase list to the phrase matcher

phrase_matcher.add('AI', None, *patterns)   #name of matcher is AI

#Apply matcher to doc
#Like rule-based matching, we again need to apply our phrase matcher to the document. However, our parsed article is not in spaCy document format. 
#Therefore, we will convert our article into sPacy document format and will then apply our phrase matcher to the article.

document = nlp(extracted_text)

matched_phrases = phrase_matcher(document)

#To see the string value of the matched phrases

for match_id, start, end in matched_phrases:
    string_id = nlp.vocab.strings[match_id]  
    span = document[start:end]                   
    print(match_id, string_id, start, end, span.text)
    

#Using Regex to get following values
    
import re
string1= "Water Depth"

regex= re.compile(r"\w+(?=\b)")

matches= re.findall(regex, string1)

for phrases in extracted_text:
        print(phrases)

w_depth = re.findall(r"([^.]*?ft.[^.]*\.)",extracted_text)                                                                                                                             

'''
'''
words = ['Water Depth']
output = []
sentences = re.findall(r"([^.]*\.)" ,extracted_text)  
for sentence in sentences:
    if all(word in sentence for word in words):
        output.append(sentence)'''

#Function to find the 
def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

x = find(txt, 'ft')

y = range(1,10)
