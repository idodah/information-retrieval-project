from flask import Flask, request, jsonify
import pickle
import csv
import gzip
from collections import Counter
from inverted_index_gcp import *
from google.cloud import storage

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    storage_client = storage.Client()
    bucket = storage_client.bucket('316608348-1')
    blob = bucket.blob('search_postings_gcp/search_index.pkl')
    pickle_in = blob.download_as_string()
    data = pickle.loads(pickle_in)
    tokens = query.split(" ")
    inverted = InvertedIndex()
    for i in tokens:
        inverted.posting_locs[i] = data.posting_locs[i]
        inverted.df[i] = data.df[i]
    df = []
    for i in inverted.posting_lists_iter():
        for j in i[1]:
            df.append(j[0])

    c = Counter(df).most_common(100)

    infile = open('id_title.pkl', 'rb')
    reader = pickle.load(infile)

    for i in c:
        if (i[0] in reader):
            res.append((i[0], reader[i[0]]))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    storage_client = storage.Client()
    bucket = storage_client.bucket('316608348-1')
    blob = bucket.blob('title_postings_gcp/title_index.pkl')
    pickle_in = blob.download_as_string()
    data = pickle.loads(pickle_in)
    tokens = query.split(" ")
    inverted = InvertedIndex()
    for i in tokens:
        inverted.posting_locs[i] = data.posting_locs[i]
        inverted.df[i] = data.df[i]
    df = []
    for i in inverted.posting_lists_iter():
        for j in i[1]:
            df.append(j[0])

    c = Counter(df).most_common(100)

    infile = open('id_title.pkl', 'rb')
    reader = pickle.load(infile)

    for i in c:
        a = reader[i[0]]
        res.append((i[0], a))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    storage_client = storage.Client()
    bucket = storage_client.bucket('316608348-1')
    blob = bucket.blob('anchor_postings_gcp/anchor_index.pkl')
    pickle_in = blob.download_as_string()
    data = pickle.loads(pickle_in)
    tokens = query.split(" ")
    inverted = InvertedIndex()
    for i in tokens:
        inverted.posting_locs[i] = data.posting_locs[i]
        inverted.df[i] = data.df[i]
    df = []
    for i in inverted.posting_lists_iter():
        for j in i[1]:
            df.append(j[0])

    c = Counter(df).most_common(100)

    infile = open('id_title.pkl', 'rb')
    reader = pickle.load(infile)

    for i in c:
        if (i[0] in reader):
            a = reader[i[0]]
            res.append((i[0], a))
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    csvFile = gzip.open('pr_part-00000-c90511d4-4864-435c-98ff-ee68c2219abd-c000.csv.gz', 'rt',
                        newline='')
    reader = csv.reader(csvFile)

    mydict = dict((rows[0], rows[1]) for rows in reader)
    for i in wiki_ids:
        res.append(float(mydict[str(i)]))
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    with open('pageviews-202108-user.pkl', 'rb') as f:
        wid2pv = pickle.loads(f.read())
    for i in wiki_ids:
        res.append(int(wid2pv[i]))
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)

