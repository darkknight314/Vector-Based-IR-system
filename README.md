# Vector-Based-IR-system
Vector-based Information Retrieval System
The assignment was completed as part of the course CS F469 Information Retrieval. The aim of the assignment is to build a vector space based information retrieval system. It was programmed using the Python NLTK library. The assignment is divided in two parts. The first part is a ranked retrieval system built using the lnc.ltc scoring scheme. The second part is the implementation of two different self-proposed improvements to the first part to overcome some flaw or limitation. The two proposed implementations are:
1.  Text normalisation on the query: This would apply case folding (converting everything to lowercase), followed by spell correction (based on the corpus vocabulary), followed by lemmatizing the tokens. Also, while computing cosine similarity, only high idf terms have been taken into account.
2. Hybrid approach to add a bigram index weighting for the K(=1000) most frequent collocations (weighted by chi-square scores after case folding) to the naive unigram weighting model. Whenever a query is taken as input, we compute the score of queries as a linear combination of their unigram weight and their bigram weight. It must be noted unigram weights are computed for all the terms in the dictionary while bigram weights are computed for only the top K bigrams. The formula for the score is score = w1*unigramWt + w2*bigramWt, where w1+w2=1.

The folders corresponding to part1,modification 1 and modification 2 are Part 1, Part 2 Improvement 1 and Part 2 Improvement 2 respectively. The corpus used is a large set of Wikipedia pages which is named wiki_47.


To run a model follow these steps:-
1. Open command prompt
2. cd into directory of model you wish to run
3. Run python index_creation.py to populate the required indexes from the corpus
4. Run python test_queries.py 

The report can be found here : https://github.com/darkknight314/Vector-Based-IR-system/wiki/Report.pdf
