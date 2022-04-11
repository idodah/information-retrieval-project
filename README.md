# Information-Retrieval-Project

Article search engine on Wikipedia corpus using GCP. Written in Python.                        

GCP:
We created indexes for the functions search, search_title, and search_anchor in GCP. 

For the get_pagerank and the get_pageviews functions, we used the functions from previous assignments to get the page rank and the number of views for each Wikipedia article. We later saved them as pickle files.

search_forntend:
For the functions search, search_title and search_anchor we first downloaded the index file we saved as pickle, searched for all the tokens we got in the query in this file, and found the location of each token in the bin files. We then added for each token its posting locs and its df to our local inverted index object. then, we iterated over the posting locs of the inverted index object we created and read their bin files. After that, we got the posting list of each word in the query. We took only the top 100 documents with the maximum number of words that appears in the query. for each word, we found its title by its id and returned them as a tuple.

Additional files:
'pageviews-202108-user.pkl'- a pickle file that saves for each document id it's number of views.
'pr_part-00000-c90511d4-4864-435c-98ff-ee68c2219abd-c000.csv.gz'- a compressed csv file that saves for each document id it's page rank score.
'id_title.pkl'- a pickle file that saves for each document id it's title.
