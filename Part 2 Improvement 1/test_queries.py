import nltk
nltk.download('punkt')
nltk.download('wordnet')
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

def tokenize(str,speller=False):		#lemmatize and tokenize input. if speller=True then apply spell correction
  tokens=nltk.word_tokenize(str)
  if speller:
    tokens=[wordnet_lemmatizer.lemmatize(spell.correction(token.lower())) for token in tokens]
  else:
    tokens=[wordnet_lemmatizer.lemmatize(token.lower()) for token in tokens]
  return tokens


def process_query(query_tokens,N,idf,norm,vocabulary):
    freq={}      					 			#logarithmic term frequency of query tokens present in vocabulary
    score={}     					 			#tf.idf of query
    idf_th=np.log10(2)							#Lower bound for high idf terms	
    for query_token in query_tokens:			#storing natural term frequency of query in freq only if term present in corpus 
      if query_token in vocabulary:
        if query_token in freq:
          freq[query_token]=freq[query_token]+1
        else:
          freq[query_token]=1
    sos=0										#stores sum of squares of scores for normalization
    for q in set(query_tokens):					#converting natural term frequency to logarithmic term frequency
      if q in vocabulary:
        freq[q]=(1+np.log10(freq[q]))
    for q in set(query_tokens):					#calculate tf.idf of query term only if term present in less than half the documents else set to 0
      if q in vocabulary:				
        if idf[q]>=idf_th:
          score[q]=freq[q]*idf[q]
          sos=sos+score[q]**2
        else:
          score[q]=0
    rsos=np.sqrt(sos)							#for normalization
    for q in query_tokens:
      if q in vocabulary:						
        score[q]=score[q]/rsos
    return score

def dot_prod(query_tokens,proc_query, norm, dict_docs):		#to calculate relevance of each document to query
    rank={}									#dictionary with doc_id as key and relevance as value
    for (doc, value) in dict_docs.items():
        rank[doc]=0
        for q in query_tokens:				#taking dot product of document vector and query vector
            if q in norm and doc in norm[q]:
                rank[doc]=rank[doc]+proc_query[q]*norm[q][doc]
    rel=sorted(rank.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)    	#sort in descending order of relevance
    return rel

def loadData(vocabname,normname,idfname,dictname): #load necessary info extracted from corpus using pickle
  vocab_file=open(vocabname,'rb')
  vocab=pickle.load(vocab_file)
  norm_file = open(normname,'rb')
  norm=pickle.load(norm_file)
  idf_file = open(idfname,'rb')
  idf=pickle.load(idf_file)
  dict_file = open(dictname,'rb')
  dict_docs=pickle.load(dict_file)
  vocab_file.close()
  norm_file.close()
  idf_file.close()
  dict_file.close()
  return vocab, norm, idf, dict_docs

vocabulary, norm, idf, dict_docs=loadData('./pickles/vocab','./pickles/norm','./pickles/idf', './pickles/dict_docs')

N=len(dict_docs)
doc_id=dict_docs.keys()
K=10

wordnet_lemmatizer=WordNetLemmatizer()
spell=SpellChecker()
spell.word_frequency.load_words(vocabulary)



query=input("Enter query: ")
query_tokens=tokenize(query,speller=True)
proc_query=process_query(query_tokens,N,idf,norm,vocabulary)
relevant_docs=dot_prod(query_tokens,proc_query, norm, dict_docs)
topK=relevant_docs[:K]



print("Showing results for: "," ".join(query_tokens))

i=1
for (key,value) in topK:
    title=dict_docs[key]
    print("Document",i)
    i=i+1
    print(key,title,value)
    print("")