import numpy as np
import tensorflow as tf
import chakin
import gensim
from copy import deepcopy
import re
import json
from random import shuffle

# Text cleaning function
def clean_text( text ):
	text = re.sub(r"\,", "", text)
	text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
	text = re.sub(r"what's", "what is ", text)
	text = re.sub(r"\'s", " ", text)
	text = re.sub(r"\'ve", " have ", text)
	text = re.sub(r"can't", "cannot ", text)
	text = re.sub(r"n't", " not ", text)
	text = re.sub(r"i'm", "i am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)
	text = re.sub(r",", " ", text)
	text = re.sub(r";", "", text)
	text = re.sub(r"\.", "", text)
	text = re.sub(r"!", " ", text)
	text = re.sub(r"\/", " ", text)
	text = re.sub(r"\^", " ", text)
	text = re.sub(r"\+", " ", text)
	text = re.sub(r"\-", " ", text)
	text = re.sub(r"\=", " ", text)
	text = re.sub(r"'", " ", text)
	text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
	text = re.sub(r":", " ", text)
	text = re.sub(r" e g ", "for example", text)
	text = re.sub(r"e - mail", "email", text)
	text = re.sub(r"\s{2,}", " ", text)
	return text


#------------quotations

# FIRST SET - JSON FILE - quotes.json
# https://raw.githubusercontent.com/4skinSkywalker/Database-Quotes-JSON/master/quotes.json
data = json.load( open('quotes.json', 'r') )
 
quotes_json = []
for i in range( len(data) ):
	temp = data[i]['quoteText'].lower().split('.')
	if temp[ len(temp)-1 ] == '': del(temp[len(temp)-1]) # remove trailing '' from the split. 
	temp = [ clean_text(i) for i in temp ]
	temp = [ i.split() for i in temp ]
	for i in temp: quotes_json.append(i)


# SECOND SET - JSON FILE - other_quotes.json
# https://gist.githubusercontent.com/nasrulhazim/54b659e43b1035215cd0ba1d4577ee80/raw/e3c6895ce42069f0ee7e991229064f167fe8ccdc/quotes.json
data = json.load( open('other-quotes.json', 'r') )

quotes_json_2 = []
for i in range( len(data['quotes']) ):
	temp = data['quotes'][i]['quote'].lower().split('.')
	if temp[ len(temp)-1 ] == '': del(temp[len(temp)-1]) # remove trailing '' from the split. 
	temp = [ clean_text(i) for i in temp ]
	temp = [ i.split() for i in temp ]
	for i in temp: quotes_json.append(i)

quotes_json.extend( quotes_json_2 )


# THIRD SET - TEXT FILE - author-quote.txt
# https://github.com/alvations/Quotables/blob/master/author-quote.txt

text_file = open("author-quote.txt", "r")
lines = text_file.read().split("	") # In the file, quotes are split by line breaks. 
# We want to remove the author names, which are split by tabs.

# Lines is a list with each quote. At the end of each quote is '\n author_name'. So let's remove that.
quotes = [ i.split("\n")[0] for i in lines ] #done!!!

# Now to process the corpus. 
corpus_raw = [ i.lower() for i in quotes ]
corpus_raw = [ clean_text(i) for i in corpus_raw ]
sentences = [ i.split() for i in corpus_raw ]
sentences.extend(quotes_json)
sentences.sort( key = lambda x: len(x) )
sentences = [ x for x in sentences if x != [] ]

# Tokenize into words 'manually', create wordset and method of
# turning words into integers/indices.
words = []
for sentence in sentences:
	tempwords = []
	for word in sentence:
		tempwords.append(word)
	words.extend(tempwords)

words = list( set(words) ) # A 'set' in Python is an unordered collection of unique elements. 
# NOTES: 
# - Set objects do not support indexing. 
# - Might be used for membership testing, elimination of duplicate elements. 
# - Set comprehension is also supported ( e.g., {x for x in 'derp' if x not in 'burbadurb'} returns {'p', 'e'} )

words.insert(0,"PAD") # Add the ID for our padding value (zero).
words.insert(1,"<GO>") # Add the ID for our start-of-sentence token.
words.insert(2,"<EOS>")

word2int = {}
int2word = {}
vocab_size = len(words)
# NOTES:
# - enumerate returns a list of tuples (index, item).
# - The list might be (0, 'if'), (1, 'to'), (2, 'from'), and so on. 
for i,word in enumerate(words):
	word2int[word] = i
	int2word[i] = word

sentences = [ i.split() for i in corpus_raw ]
sentences.extend(quotes_json)
sentences.sort( key = lambda x: len(x) )

sentences = [ x[::-1] for x in sentences if x != [] ] # Sentence reversal 'trick'.

# Create word ID dataset
word_ids = []
for sentence in sentences:
	temp = [ word2int[i] for i in sentence ]
	word_ids.append( temp )

# Data structuring
def generate_batches( inputs, batch_size ):
	counter = 0
	batches = []
	batches_len = []
	while counter < len(inputs):
		currbatch = inputs[ counter : min( counter+batch_size, len(inputs) ) ]
		currbatch, currbatch_len = batch( currbatch )
		batches.append(currbatch)
		batches_len.append( currbatch_len)
		counter = counter+batch_size
	return batches, batches_len

def batch( inputs, max_seq_len = None ):
    seq_len = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    if max_seq_len is None:
        max_seq_len = max(seq_len) 
    inputs_batch_major = np.zeros(shape=[batch_size, max_seq_len], dtype=np.int32) # == PAD 
    for i, seq in enumerate(inputs):
        pad = np.zeros( ( max_seq_len - len(seq ) ), dtype = np.int32 )
        insert = np.concatenate( (seq, pad) )
        inputs_batch_major[i] = insert
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)
    return inputs_time_major, seq_len

inputs = deepcopy(word_ids)


