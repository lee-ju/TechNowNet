import os
import re
import time
import argparse
import pandas as pd
import gensim
from multiprocessing import Pool
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS
from utils import download_nltk_resources, tag2wordnettag
NON_ENG_PATTERN = re.compile(r'[\uAC00-\uD7AF\u3040-\u30FF\u4E00-\u9FFF]')

def refine_tokens(text):
	tokens = word_tokenize(text.lower())
	refined_tokens = [t.replace(' ', '') for t in tokens if len(t.replace(' ', '')) > 1]
	return refined_tokens

def process_unit(key_value_pair):
	period, texts = key_value_pair
	lof_tokens = [refine_tokens(text) for text in texts]   
	return period, lof_tokens     

def tokenize(directory_path, aggregation_period, n_cpus):
	print(f"\n[Step 1] Tokenizing files in {directory_path} (By {aggregation_period})")
	flocs = [os.path.join(p, n) for p, s, files in os.walk(directory_path) for n in files if '.filt' in n and '.tok' not in n]
	
	for floc in sorted(flocs):
		df = pd.read_csv(floc, sep='\t')
		period2texts = defaultdict(list)
		
		for (year, period_val), subdf in df.groupby(['year', aggregation_period]):
			p_str = str(year) + str(period_val).zfill(2)
			texts = [str(t) + ' ' + str(s) for t, s in subdf[['title', 'summary']].values]
			period2texts[p_str] += texts

		period2tokens = defaultdict(list)
		with Pool(processes=n_cpus) as pool:
			for period, lof_tokens in pool.imap(process_unit, period2texts.items()):
				period2tokens[period] += lof_tokens

		for period, lof_tokens in period2tokens.items():
			out_dir = os.path.join(directory_path, period[:4])
			os.makedirs(out_dir, exist_ok=True)
			out_path = os.path.join(out_dir, f"{period}.tok")
			with open(out_path, 'a', encoding='utf-8') as f:
				for tokens in lof_tokens:
					f.write('\t'.join(tokens) + '\n')

def ngram(directory_path, repeat_n):
	print(f"[Step 2] N-gram processing (Repeat: {repeat_n})")
	flocs = [os.path.join(p, n) for p, s, files in os.walk(directory_path) for n in files if '.tok' in n and '.ngram' not in n]
	
	for floc in sorted(flocs):
		with open(floc, 'r', encoding='utf-8') as f:
			lof_tokens = [line.strip().split('\t') for line in f]
		
		for _ in range(repeat_n):
			model = gensim.models.Phrases(lof_tokens, delimiter=' ', connector_words=ENGLISH_CONNECTOR_WORDS)
			lof_tokens = [model[t] for t in lof_tokens]
			
		with open(floc + '.ngram', 'w', encoding='utf-8') as f:
			for tokens in lof_tokens:
				f.write('\t'.join(tokens) + '\n')

def lemma_worker(line):
	text_tokens = line.strip().split('\t')
	stopws = set(stopwords.words('english'))
	lemmatizer = WordNetLemmatizer()
	
	lemma_tokens = []
	for w, pos in pos_tag(text_tokens):
		if w in stopws or len(w) < 2: continue
		npos = tag2wordnettag(pos)
		lemma_tokens.append(lemmatizer.lemmatize(w, npos) if npos else w)
	return lemma_tokens

def lemmatize(directory_path, n_cpus):
	print(f"[Step 3] Lemmatizing N-gram files")
	flocs = [os.path.join(p, n) for p, s, files in os.walk(directory_path) for n in files if '.ngram' in n and '.lem' not in n]
	
	for floc in sorted(flocs):
		with open(floc, 'r', encoding='utf-8') as f:
			lines = f.readlines()
		
		with open(floc + '.lem', 'w', encoding='utf-8') as f:
			with Pool(processes=n_cpus) as pool:
				for lem_tokens in pool.imap(lemma_worker, lines):
					f.write('\t'.join(lem_tokens) + '\n')

def merge(directory_path, year_start, year_end, name):
	print(f"\n[Step 4] Merging into {name} (Intermediate format)")
	path = os.path.join(directory_path, name)
	
	with open(path, 'w', encoding='utf-8') as w:
		for year in range(year_start, year_end + 1):
			year_dir = os.path.join(directory_path, str(year))
			if not os.path.exists(year_dir): continue
			
			for file in sorted(os.listdir(year_dir)):
				if file.endswith('.tok.ngram.lem'):
					period = file.split('.')[0]
					with open(os.path.join(year_dir, file), 'r', encoding='utf-8') as r:
						for line in r:
							tokens = line.strip().split('\t')
							if tokens:
								w.write(f"{period}<SEP>{'\t'.join(tokens)}\n")
	print(f"Processing Complete: {path}")

if __name__ == '__main__':
	download_nltk_resources()
	
	parser = argparse.ArgumentParser(description="Integrated News Preprocessing Pipeline (TechNowNet)")
	parser.get_default('--year_start')
	parser.add_argument('--year_start', type=int, default=2019)
	parser.add_argument('--year_end', type=int, default=2023)
	parser.add_argument('--ngram', type=int, default=4)
	parser.add_argument('--n_cpus', type=int, default=4)
	parser.add_argument('--wdir', type=str, default='./dataset/mas2lem/') # Market-and-Society: mas

	args = parser.parse_args()

	tokenize(args.wdir, 'year', args.n_cpus)
	ngram(args.wdir, args.ngram)
	lemmatize(args.wdir, args.n_cpus)
	filename = f"mas.{args.year_start}-{args.year_end}.lem.docu"
	merge(args.wdir, args.year_start, args.year_end, filename)