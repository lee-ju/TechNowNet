import os
import time
import re
import argparse
import pandas as pd
import numpy as np
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Phrases
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS
from utils import download_nltk_resources, tag2wordnettag

def refine_tokens(text):
	tokens = word_tokenize(text)
	refined_tokens = []
	for token in tokens:
		token = token.lower()
		token = re.sub(' ', '', token)
		if len(token) >= 1:
			refined_tokens.append(token)
	return refined_tokens

def tokenization(year, yinfo_dir, output_dir, n_cpus):
	print(f'\n[Step 1] Tokenizing Year: {year}')
	input_path = os.path.join(yinfo_dir, f'tech.{year}.pat')
	output_path = os.path.join(output_dir, f'tech.{year}.tok')
	
	patstat = pd.read_csv(input_path, sep='\t', encoding='utf-8')
	patstat['text'] = patstat['appln_title'].fillna('') + '. ' + patstat['appln_abstract'].fillna('')
	patstat.dropna(subset=['text'], inplace=True)
	texts = patstat['text'].to_list()

	with open(output_path, 'w', encoding='utf-8') as f:
		with Pool(n_cpus) as p:
			for tokens in p.imap(refine_tokens, texts):
				f.write(f"{year}<SEP>{'	'.join(tokens)}\n")
	return output_path

def ngram(year, output_dir, ngram_depth):
	print(f'[Step 2] Generating N-grams Year: {year}')
	input_path = os.path.join(output_dir, f'tech.{year}.tok')
	output_path = os.path.join(output_dir, f'tech.{year}.tok.ngram')

	with open(input_path, 'r', encoding='utf-8') as r:
		lines = r.readlines()
	tokens = [line.strip().split('<SEP>')[1].split('\t') for line in lines]

	for _ in range(ngram_depth):
		model = Phrases(tokens, delimiter='_', connector_words=ENGLISH_CONNECTOR_WORDS)
		tokens = [model[token] for token in tokens]

	with open(output_path, 'w', encoding='utf-8') as w:
		for token_list in tokens:
			w.write(f"{year}<SEP>{'	'.join(token_list)}\n")
	return output_path

def lemma_worker(line):
	year, text_tokens = line.strip().split('<SEP>')
	text_tokens = text_tokens.split('\t')
	stopws = set(stopwords.words('english'))
	lemmatizer = WordNetLemmatizer()
	
	lemma_text_tokens = []
	for w, pos in pos_tag(text_tokens):
		if w in stopws or len(w) < 2:
			continue
		npos = tag2wordnettag(pos)
		if npos is None:
			lemma_text_tokens.append(w)
		else:
			lemma_text_tokens.append(lemmatizer.lemmatize(w, npos))
	return year, lemma_text_tokens

def lemmatization(year, output_dir, n_cpus):
	print(f'[Step 3] Lemmatizing Year: {year}')
	input_path = os.path.join(output_dir, f'tech.{year}.tok.ngram')
	output_path = os.path.join(output_dir, f'tech.{year}.tok.ngram.lem')

	with open(input_path, 'r', encoding='utf-8') as r:
		lines = r.readlines()

	with open(output_path, 'w', encoding='utf-8') as w:
		with Pool(n_cpus) as p:
			for year_val, lems in p.imap(lemma_worker, lines):
				w.write(f"{year_val}<SEP>{'	'.join(lems)}\n")
	return output_path

def proc(line):
	date, tokens = line.strip().split('<SEP>')
	tokens = tokens.split('\t')
	tokens = [re.sub(' ', '_', token) for token in tokens]
	return f"{date}<SEP>{'	'.join(tokens)}\n"

def merge(year_start, year_end, output_dir, filename, n_cpus):
	print(f'\n[Step 4] Merging all years into {filename}')
	path = os.path.join(output_dir, filename)
	
	with open(path, 'w', encoding='utf-8') as w:
		for year in range(year_start, year_end + 1):
			input_path = os.path.join(output_dir, f'tech.{year}.tok.ngram.lem')
			if not os.path.exists(input_path): continue
			
			with open(input_path, 'r', encoding='utf-8') as r:
				lines = r.readlines()
			
			with Pool(n_cpus) as p:
				for processed_line in p.imap(proc, lines):
					w.write(processed_line)
	print(f"All process finished. Final file: {path}")

if __name__ == '__main__':
	download_nltk_resources()
	
	parser = argparse.ArgumentParser(description="Integrated Patent Preprocessing Pipeline (TechNowNet)")
	parser.add_argument('--year_start', type=int, default=2019)
	parser.add_argument('--year_end', type=int, default=2023)
	parser.add_argument('--n_cpus', type=int, default=20)
	parser.add_argument('--ngram', type=int, default=4)
	parser.add_argument('--wdir', type=str, default='./data/align/')
	parser.add_argument('--yinfo', type=str, default='./data/tech/yinfo/')
	
	args = parser.parse_args()
	
	output_base = os.path.join(args.wdir, 'dataset', 'pat2lem')
	os.makedirs(output_base, exist_ok=True)

	for current_year in range(args.year_start, args.year_end + 1):
		tokenization(current_year, args.yinfo, output_base, args.n_cpus)
		ngram(current_year, output_base, args.ngram)
		lemmatization(current_year, output_base, args.n_cpus)

	name = f"tech.{args.year_start}-{args.year_end}.lem"
	merge(args.year_start, args.year_end, output_base, name, args.n_cpus)