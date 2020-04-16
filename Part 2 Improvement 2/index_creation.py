from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
import numpy as np
import pickle
from nltk.util import ngrams
from collections import Counter

INDEX="./pickles/"

file_content = open("wiki_47",encoding="utf8").read()
all_docs = file_content.split("</doc>")            #Soup was originally utilised to retieve individual documents but it resulted in incorrect partitioning of documents
all_docs = [BeautifulSoup(doc+"</doc>", "lxml") for doc in all_docs][:-1]

def tokenize(str):
  tokens=nltk.word_tokenize(str)
  tokens=[token.lower() for token in tokens]
  return tokens

#Creation of list of doc ids, doc titles and document text to zip them together
doc_id = []
doc_title = []
doc_text = []
dict_docs={}
for doc in all_docs:
    pid=doc.find_all("doc")[0].get("id")
    ptitle=doc.find_all("doc")[0].get("title")
    ptext=doc.get_text().lower()
    doc_id.append(pid)
    doc_title.append(ptitle)
    doc_text.append(ptext)
    dict_docs[pid]=ptitle
indexed_docs = list(zip(doc_id,doc_title,doc_text))

#Creation of vocabulary
tokens=[]
for page in doc_text:
  tokens.extend(tokenize(page))              
vocabulary = sorted(set(tokens))

tdf={}                  #Will store the natural term document frequencies
for term in vocabulary:
    tdf[term]={}
for doc_iter in indexed_docs:
    dc_id=doc_iter[0]
    doc_tokens=tokenize(doc_iter[2])
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

bigrams=[]					#list of all bigrams in corpus
bigram_frequency = {}		#frequency of bigrams
first_word = {}             #Frequency of unigrams
second_word = {}            #Frequency of unigrams

for text in doc_text:		#fill bigrams
  temp=list(ngrams(tokenize(text),2))
  bigrams.extend(temp)

unique_bigrams = list(set(bigrams))

total_bigrams = len(bigrams)

for bi in bigrams:						#fill bigram_frequency
  if bi in bigram_frequency:
    bigram_frequency[bi]=bigram_frequency[bi]+1
  else:
    bigram_frequency[bi]=1

for x in tokens:						
    first_word[x] = 0
    second_word[x] = 0
for x in tokens:						#fill first_word and second_word
    first_word[x] = first_word[x] + 1
    second_word[x] = second_word[x] + 1

chi_square_scores = {}
for bigram in unique_bigrams:			#calculate chi-square scores for all bigrams 
	word1 = bigram[0]
	word2 = bigram[1]
	o11 = bigram_frequency[bigram]
	o21 = first_word[word1] - o11
	o12 = second_word[word2] - o11
	o22 = total_bigrams - o11 - o21 - o12
	chi_score = total_bigrams*(((o11*o22-o21*o12)**2)/((o11+o21)*(o11+o12)*(o21+o22)*(o12+o22)))
	if(o21 + o12 > 10):
			chi_square_scores[bigram] = chi_score

collocations = sorted(chi_square_scores.items(), key = lambda kv:(kv[1], kv[0]),reverse=True) #sort collocations in ascending order of importance
frequent_collocations = []		#store the top 1000 collocations

count = 0
for (x,y) in collocations:
    count = count + 1
    if count <= 1000:
          frequent_collocations.append(x)
    else:
      break

#NOW WE HAVE TOP 1000 COLLOCATIONS
	  
biword_tdf ={}								
for biterm in frequent_collocations:
    biword_tdf[biterm]={}

for doc_iter in indexed_docs:               #to create natural term document frequency of frequent collocations
    dc_id = doc_iter[0]
    doc_bigrams = ngrams(tokenize(doc_iter[2]),2)
    for biword in doc_bigrams:
        if biword not in biword_tdf:
            continue
        if dc_id in biword_tdf[biword]:
            biword_tdf[biword][dc_id] = biword_tdf[biword][dc_id] + 1
        else:
            biword_tdf[biword][dc_id]=1

#to calculate bigram normalized logarithmic tf for top 1000 collocations
biword_wt={}							
biword_sos={}
for doc in doc_id:
    biword_sos[doc]=0

for biword in biword_tdf:
    biword_dicti = biword_tdf[biword]
    biword_wt[biword]={}
    for key,value in biword_dicti.items():
        biword_wt[biword][key]=1+np.log10(value)
        biword_sos[key] = biword_sos[key] + biword_wt[biword][key]**2

biword_norm={}
for biword in biword_tdf:
    biword_dicti = biword_tdf[biword]
    biword_norm[biword] = {}
    for key,value in biword_dicti.items():
        biword_norm[biword][key] = biword_wt[biword][key] / (np.sqrt(biword_sos[key]))



#Creation of index to store normalized tdf and idf values
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
bi_file = open(INDEX+'Bigram tdf','ab')
pickle.dump(biword_tdf, bi_file)
bi_file.close()
bi_norm_file = open(INDEX+'Bigram norm','ab')
pickle.dump(biword_norm, bi_norm_file)
bi_norm_file.close()
