import nltk
from nltk.util import ngrams
import numpy as np
import pickle

INDEX="./pickles/"

def tokenize(str):                  #tokenize and convert to lower case
  tokens=nltk.word_tokenize(str)
  tokens=[token.lower() for token in tokens]
  return tokens
  
def process_query(query,N,norm,vocabulary):         #to calculate normalized tf.idf of query
    freq={}                                         #logarithmic term frequency of query tokens present in vocabulary
    score={}     							                      #tf.idf of query
    query_tokens=tokenize(query)
    for query_token in query_tokens:                #storing natural term frequency of query in freq only if term present in corpus
      if query_token in vocabulary:
        if query_token in freq:
          freq[query_token]=freq[query_token]+1
        else:
          freq[query_token]=1
    sos=0
    for q in set(query_tokens):                     #converting natural term frequency to logarithmic term frequency
      if q in vocabulary:
        freq[q]=(1+np.log10(freq[q]))
    for q in set(query_tokens):                     #calculating tf.idf score of query vector and sum of squares for cosine normalization
      if q in vocabulary:
        score[q]=freq[q]*idf[q]
        sos=sos+score[q]**2
      else:
        score[q] = 0
    rsos=np.sqrt(sos)                                #to normalize query vector
    for q in query_tokens:	                  			 #normalization of query vector
      if q in vocabulary:
        score[q]=score[q]/rsos
    return score


def process_bigrams(query, N, biword_tdf):            #calculate tf.idf score for bigrams in query(only for bigrams in top 1000)
    freq={}                                           #frequency vector of top bigrams in query
    idf={}                                            #inverse document frequency of each bigram
    score={}                                          #
    for biword in biword_tdf:                         #fill idf and set freq to 0
        freq[biword]=0
        if len(biword_tdf[biword])==0:
            idf[biword]=0
        else:
            idf[biword]=np.log10(N/len(biword_tdf[biword]))
    query_tokens_list = tokenize(query)
    query_tokens = list(ngrams(query_tokens_list,2))   #bigrams in query
    
    for query_token in query_tokens:                   #fill freq vector
        if query_token in biword_tdf:
            freq[query_token] = freq[query_token] + 1
    sos=0
    for q in set(query_tokens):                         #convert natural frequency to logarthmic frequency
        if q in biword_tdf:
            freq[q] = (1+np.log10(freq[q]))
    for q in set(query_tokens):                         
        if q in biword_tdf:
            score[q]=freq[q]*idf[q]                     #tf.idf score of query bigrams(only for bigrams in top 1000)
            sos=sos+score[q]**2                         #sum of squares for normalization
    rsos=np.sqrt(sos)         
    for q in query_tokens:
        if q in biword_tdf:
            score[q]=score[q]/rsos
    return score

def dot_prod(query,proc_query,proc_bigrams, norm, dict_docs, biword_norm):      #calculate ranking system of documents for query. Score=unigram_score+0.9*bigram_score
    query_tokens=tokenize(query)
    query_bigrams = list(ngrams(query_tokens,2))
    rank={}
    for (doc,value) in dict_docs.items():                                   
        rank[doc]=0
        for q in query_tokens:                           #add unigram tf.idf score
            if q in norm and doc in norm[q]:
                rank[doc]=rank[doc]+proc_query[q]*norm[q][doc]
        rank[doc]=rank[doc]
        for q in query_bigrams:                          #add bigram tf.idf score 
            if q in biword_norm and doc in biword_norm[q]:
                rank[doc] = rank[doc] + 0.9*proc_bigrams[q]*biword_norm[q][doc]
    rel=sorted(rank.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)    
    return rel

def loadData(normname,idfname,dictname,biname,vocabname, binormname):
	norm_file = open(INDEX+normname,'rb')
	norm=pickle.load(norm_file)
	idf_file = open(INDEX+idfname,'rb')
	idf=pickle.load(idf_file)
	dict_file = open(INDEX+dictname,'rb')
	dict_docs=pickle.load(dict_file)
	bi_file = open(INDEX+biname,'rb')
	biword_tdf=pickle.load(bi_file)
	vocabulary_file = open(INDEX+vocabname,'rb')
	vocabulary=pickle.load(vocabulary_file)
	bi_norm_file = open(INDEX+binormname,'rb')
	biword_norm=pickle.load(bi_norm_file)
	norm_file.close()
	idf_file.close()
	dict_file.close()
	bi_file.close()
	bi_norm_file.close()
	return norm, idf, dict_docs , biword_tdf, vocabulary, biword_norm

norm, idf, dict_docs, biword_tdf, vocabulary, biword_norm=loadData('Normalized tdf','IDF', 'dict_docs','Bigram tdf','vocabulary', 'Bigram norm')

#Main code
N=len(dict_docs)
K=10
query=input("Enter query: ")
proc_query=process_query(query,N,norm,vocabulary)
query_tokens_list = tokenize(query)
query_bigrams = ngrams(query_tokens_list,2)
proc_bigrams = process_bigrams(query, N, biword_tdf)
relevant_docs=dot_prod(query,proc_query,proc_bigrams, norm, dict_docs, biword_norm)
topK=relevant_docs[:K]
query_tokens=tokenize(query)
i=1
for (key,value) in topK:
    title=dict_docs[key]
    print("Document",i)
    i=i+1
    print(key,title,value)
    print("")