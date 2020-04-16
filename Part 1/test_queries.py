import nltk
import pickle
import numpy as np

INDEX="./pickles/"

def process_query(query,N,norm,vocabulary): #to calculate normalized tf.idf of query 							
    freq={}      							#logarithmic term frequency of query tokens present in vocabulary
    score={}     							#tf.idf of query
    query_tokens=nltk.word_tokenize(query)
    for query_token in query_tokens:		#storing natural term frequency of query in freq only if term present in corpus
      if query_token in vocabulary:
        if query_token in freq:
          freq[query_token]=freq[query_token]+1
        else:
          freq[query_token]=1
    sos=0
    for q in set(query_tokens):				#converting natural term frequency to logarithmic term frequency
      if q in vocabulary:
        freq[q]=(1+np.log10(freq[q]))
    for q in set(query_tokens):				#calculating tf.idf score of query vector and sum of squares for cosine normalization
      if q in vocabulary:
        score[q]=freq[q]*idf[q]
        sos=sos+score[q]**2
    rsos=np.sqrt(sos)						#to normalize query vector
    for q in query_tokens:					#normalization of query vector
      if q in vocabulary:
        score[q]=score[q]/rsos
    return score

def dot_prod(query,proc_query, norm, dict_docs):		#to calculate relevance of each document to query
    query_tokens=nltk.word_tokenize(query)
    rank={}									#dictionary with doc_id as key and relevance as value
    for (doc, value) in dict_docs.items():
        rank[doc]=0
        for q in query_tokens:				#taking dot product of document vector and query vector
            if q in norm and doc in norm[q]:
                rank[doc]=rank[doc]+proc_query[q]*norm[q][doc]
    rel=sorted(rank.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)    	#sort in descending order of relevance
    return rel

def loadData(normname,idfname,dictname, vocabname):	#load necessary info extracted from corpus using pickle
	norm_file = open(INDEX+normname,'rb')
	norm=pickle.load(norm_file)
	idf_file = open(INDEX+idfname,'rb')
	idf=pickle.load(idf_file)
	dict_file = open(INDEX+dictname,'rb')
	dict_docs=pickle.load(dict_file)
	vocabulary_file = open(INDEX+vocabname,'rb')
	vocabulary=pickle.load(vocabulary_file)
	return norm, idf, dict_docs, vocabulary

#Main code
norm, idf, dict_docs, vocabulary=loadData('Normalized tdf', 'IDF', 'dict_docs', 'vocabulary')
N=len(dict_docs)							#number of documents
K=10										#top K documents to be returned
query=input("Enter query: ")
proc_query=process_query(query,N,norm,vocabulary)
print()
relevant_docs=dot_prod(query, proc_query, norm, dict_docs)
topK=relevant_docs[:K]						#select top K documents out of ranked documents
i=1
for (key,value) in topK:
    title=dict_docs[key]
    print("Document",i)
    i=i+1
    print(key,title,value)
    print("")