import os
import time
import re
import argparse
import gensim
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS
from utils import download_nltk_resources, tag2wordnettag

def refine_tokens(tokens, is_authkwds=False):
	refined_tokens = []
	for token in tokens:
		token = token.lower()
		if is_authkwds:
			token = ' '.join(token.split())
			token = token.strip().strip('"').strip('.').strip(',')
		else:
			token = token.replace(' ', '')
		if len(token) > 1:
			refined_tokens.append(token)
	return refined_tokens

def tokenize_worker(line):
	parts = line.strip().split('\t')
	if len(parts) < 5: return None
	eid, year, text, authkwds, asjcs = parts[0], parts[1], parts[2], parts[3], parts[4]

	text_tokens = word_tokenize(text.lower())
	text_tokens = refine_tokens(text_tokens, is_authkwds=False)

	authkwds_tokens = authkwds.lower().split(';')
	authkwds_tokens = refine_tokens(authkwds_tokens, is_authkwds=True)
	
	asjcs_list = asjcs.split(';')

	return year, text_tokens, authkwds_tokens, asjcs_list

def tokenize(input_file, output_dir, n_cpus, chunk_size):
	print(f"\n[Step 1] Tokenizing: {input_file}")
	if not os.path.exists(output_dir): os.makedirs(output_dir)
	
	with open(input_file, 'r', encoding='utf-8') as f:
		lines = f.readlines()[1:]

	years_found = set()
	file_handles = {}

	with Pool(processes=n_cpus) as pool:
		for result in pool.imap(tokenize_worker, lines, chunksize=chunk_size):
			if result is None: continue
			year, text_toks, auth_toks, asjcs = result
			
			if year not in years_found:
				file_path = os.path.join(output_dir, f"sci.{year}.tok")
				file_handles[year] = open(file_path, "w", encoding="utf-8")
				years_found.add(year)
			
			sent = '\t'.join(text_toks) + '<SEP>' + '\t'.join(auth_toks) + '<SEP>' + '\t'.join(asjcs) + '\n'
			file_handles[year].write(sent)
			
	for h in file_handles.values(): h.close()

def ngram(year, output_dir, ngram_depth):
	print(f"[Step 2] N-gram Processing Year: {year}")
	input_path = os.path.join(output_dir, f"sci.{year}.tok")
	output_path = os.path.join(output_dir, f"sci.{year}.tok.ngram")
	
	if not os.path.exists(input_path): return

	text_list, auth_list, asjc_list = [], [], []
	with open(input_path, 'r', encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split('<SEP>')
			text_list.append(parts[0].split('\t'))
			auth_list.append(parts[1].split('\t'))
			asjc_list.append(parts[2].split('\t'))

	for _ in range(ngram_depth):
		model = gensim.models.Phrases(text_list, delimiter=' ', connector_words=ENGLISH_CONNECTOR_WORDS)
		text_list = [model[doc] for doc in text_list]

	with open(output_path, 'w', encoding='utf-8') as f:
		for t, a, j in zip(text_list, auth_list, asjc_list):
			f.write(f"{year}<SEP>{'\t'.join(t)}<SEP>{'\t'.join(a)}<SEP>{'\t'.join(j)}\n")

def lemma_worker(line):
	parts = line.strip().split('<SEP>')
	year, text_tokens, auth_tokens, asjcs = parts[0], parts[1].split('\t'), parts[2].split('\t'), parts[3].split('\t')
	
	stopws = set(stopwords.words('english'))
	lemmatizer = WordNetLemmatizer()
	
	lemma_text = []
	for w, pos in pos_tag(text_tokens):
		if w in stopws or len(w) < 2: continue
		npos = tag2wordnettag(pos)
		lemma_text.append(lemmatizer.lemmatize(w, npos) if npos else w)

	lemma_auth = [lemmatizer.lemmatize(w, wordnet.NOUN) for w in auth_tokens if w not in stopws and len(w) >= 2]

	return year, lemma_text, lemma_auth, asjcs

def lemmatize(year, output_dir, n_cpus):
	print(f"[Step 3] Lemmatizing Year: {year}")
	input_path = os.path.join(output_dir, f"sci.{year}.tok.ngram")
	output_path = input_path + ".lem"
	
	if not os.path.exists(input_path): return

	with open(input_path, 'r', encoding='utf-8') as f:
		lines = f.readlines()

	with open(output_path, 'w', encoding='utf-8') as f:
		with Pool(n_cpus) as p:
			for y, l_text, l_auth, asjcs in p.imap(lemma_worker, lines):
				f.write(f"{y}<SEP>{'\t'.join(l_text)}<SEP>{'\t'.join(l_auth)}<SEP>{'\t'.join(asjcs)}\n")

def proc(line):
	parts = line.strip().split('<SEP>')
	year, tokens, authkwds, asjcs = parts[0], parts[1].split('\t'), parts[2], parts[3]
	tokens = [re.sub(' ', '_', t) for t in tokens]
	return f"{year}<SEP>{'\t'.join(tokens)}<SEP>{authkwds}<SEP>{asjcs}\n"

def merge(year_start, year_end, output_dir, filename, n_cpus):
	print(f"\n[Step 4] Merging Scopus data: {filename}")
	path = os.path.join(output_dir, filename)
	
	with open(path, 'w', encoding='utf-8') as w:
		for year in range(year_start, year_end + 1):
			input_path = os.path.join(output_dir, f"sci.{year}.tok.ngram.lem")
			if not os.path.exists(input_path): continue
			
			with open(input_path, 'r', encoding='utf-8') as r:
				lines = r.readlines()
			
			with Pool(n_cpus) as p:
				for processed in p.imap(proc, lines):
					w.write(processed)
	print(f"All Scopus processing finished. File: {path}")

if __name__ == '__main__':
	download_nltk_resources()
	
	parser = argparse.ArgumentParser(description="Integrated Scopus Preprocessing Pipeline for TechNowNet")
	parser.add_argument('--year_start', type=int, default=2019)
	parser.add_argument('--year_end', type=int, default=2023)
	parser.add_argument('--n_cpus', type=int, default=20)
	parser.add_argument('--chunk_size', type=int, default=100)
	parser.add_argument('--ngram', type=int, default=4)
	parser.add_argument('--wdir', type=str, default='./data/align/')
	parser.add_argument('--input_file', type=str, default='sci.refined')

	args = parser.parse_args()
	yinfo_dir = os.path.join(args.wdir, 'yinfo')

	tokenize(args.input_file, yinfo_dir, args.n_cpus, args.chunk_size)

	for y in range(args.year_start, args.year_end + 1):
		ngram(y, yinfo_dir, args.ngram)
		lemmatize(y, yinfo_dir, args.n_cpus)

	name = f"sci.{args.year_start}-{args.year_end}.lem"
	merge(args.year_start, args.year_end, args.wdir, name, args.n_cpus)