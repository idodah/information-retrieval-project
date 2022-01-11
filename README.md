# information-retrieval-project
gcp:
we created indexes for the functions search, search_title and search_anchor in gcp. 

for the get_pagerank and the get_pageviews functions we used the functions from previous assignments to get the page rank and the number of views for each wikipedia article. we later saved them as csv.gz and pkl files.

search_forntend:
for the functions search, search_title and search_anchor we first downloaded the index file we saved as pkl, searched for all the tokens we got in the query in this file and found the location of each token in the bin files. we then added for each token it's posting locs and it's df to our local inverted index object. then, we iterated over the posting locs of the inverted index object we created and read their bin files. after that we got the posting list of each word in the query. we took only the top 100 documents with the maximum number of words that appears in the query. for each word, we found its title by its id and returned them as a tuple.

for the get_pagerank and the get_pageviews functions we downloaded files that saves for each document the pagerank score and the number of page views it got.

additional files:
'pageviews-202108-user.pkl'- a pickle file that saves for each document id it's number of views.
'pr_part-00000-c90511d4-4864-435c-98ff-ee68c2219abd-c000.csv.gz'- a compressed csv file that saves for each document id it's page rank score.
'id_title.pkl'- a pickle file that saves for each document id it's title.
