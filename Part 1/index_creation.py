from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
import numpy as np
import pickle

INDEX="./pickles/"

file_content = open("wiki_47",encoding="utf8").read()
all_docs = file_content.split("</doc>")            #Soup was originally utilised to retieve individual documents but it resulted in incorrect partitioning of documents
all_docs = [BeautifulSoup(doc+"</doc>", "lxml") for doc in all_docs][:-1]

#Creation of list of doc ids, doc titles and document text to zip them together
doc_id = []
doc_title = []
doc_text = []
dict_docs={}
for doc in all_docs:
    pid=doc.find_all("doc")[0].get("id")
    ptitle=doc.find_all("doc")[0].get("title")
    ptext=doc.get_text()
    doc_id.append(pid)
    doc_title.append(ptitle)
    doc_text.append(ptext)
    dict_docs[pid]=ptitle
indexed_docs = list(zip(doc_id,doc_title,doc_text))

#Creation of vocabulary
tokens=[]
for page in doc_text:
  tokens.extend(nltk.word_tokenize(page))              
vocabulary = sorted(set(tokens))

tdf={}                  #Will store the natural term document frequencies
for term in vocabulary:
    tdf[term]={}
for doc_iter in indexed_docs:
    dc_id=doc_iter[0]
    doc_tokens=nltk.word_tokenize(doc_iter[2])
    for term in doc_tokens:
        if term in tdf:
            if dc_id in tdf[term]:
                tdf[term][dc_id]=tdf[term][dc_id]+1
            else:
                tdf[term][dc_id]=1
            
wt={}                    #Will store the logarithmically scaled term documnet frequencies
sos={}                   #Sum of squares of logarithmic term document frequencies for normalization
for doc in doc_id:
    sos[doc]=0
for term in vocabulary:
    dicti=tdf[term]
    wt[term]={}
    for key,value in dicti.items():
        wt[term][key]=1+np.log10(value)
        sos[key]=sos[key]+wt[term][key]**2

norm={}                   #Normalized logarithmic term document frequencies
for term in vocabulary:
    dicti=tdf[term]
    norm[term]={}
    for key,value in dicti.items():
        norm[term][key]=wt[term][key]/(np.sqrt(sos[key]))

idf={}        #inverse document frequency of dictionary
for term in vocabulary:
  if len(norm[term])==0:
    idf[term]=0
  else:
    idf[term]=np.log10(len(all_docs)/len(norm[term]))

#Creation of index
norm_file = open(INDEX+'Normalized tdf','ab')
pickle.dump(norm, norm_file)
norm_file.close()
idf_file = open(INDEX+'IDF','ab')
pickle.dump(idf, idf_file)
idf_file.close()
dict_file = open(INDEX+'dict_docs','ab')
pickle.dump(dict_docs, dict_file)
dict_file.close()
vocab_file = open(INDEX+'vocabulary','ab')
pickle.dump(vocabulary, vocab_file)
vocab_file.close()
