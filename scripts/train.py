import os
import time
import argparse
import multiprocessing
from gensim.models import FastText

class SentenceGenerator:
	def __init__(self, filepath):
		self.filepath = filepath
		self.epoch = 0
		self.train_start = time.time()

	def __iter__(self):
		if self.epoch > 0:
			elapsed = time.time() - self.train_start
			print(f"  > Completed Epoch: {self.epoch} | Cumulative Time: {elapsed:.2f}s")
		
		self.epoch += 1
		with open(self.filepath, 'r', encoding='utf-8') as f:
			for line in f:
				yield line.strip().split()

def train(wdir, db, vector_size, year_start, year_end, 
					window=5, min_count=5, sg=1, sample=1e-5, epochs=5, workers='auto'):
	start_time = time.time()
	
	input_file = f"{db}.{year_start}-{year_end}.lem.docu"
	input_path = os.path.join(wdir, 'dataset', input_file)
	
	dim_str = f"d{vector_size:03d}" if vector_size < 100 else f"d{vector_size}"
	model_name = f"{input_file}.ftx.{dim_str}"
	
	save_dir = os.path.join(wdir, 'model')
	raw_dir = os.path.join(save_dir, 'raw')
	os.makedirs(raw_dir, exist_ok=True)

	print("-" * 40)
	print(f"[*] Training Start: {model_name}")
	print(f"  - Database: {db}")
	print(f"  - Vector Size: {vector_size}")
	print(f"  - Target File: {input_path}")

	if os.path.exists(os.path.join(save_dir, model_name)):
		print(f"[!] SKIP: Model '{model_name}' already exists in {save_dir}")
		return

	if workers == 'auto':
		workers = multiprocessing.cpu_count() - 1
		
	sentences = SentenceGenerator(input_path)
	
	params = {
		'sentences': sentences,
		'vector_size': vector_size,
		'window': window,
		'min_count': min_count,
		'sg': sg,
		'sample': sample,
		'epochs': epochs,
		'workers': workers
	}

	print("  - Parameters:")
	for k, v in params.items():
		if k != 'sentences': print(f"    * {k}: {v}")

	print("\n[1/2] Training FastText model...")
	model = FastText(**params)

	print("\n[2/2] Saving model files...")
	model.save(os.path.join(raw_dir, f"{model_name}.raw"))
	
	model.wv.save_word2vec_format(os.path.join(save_dir, model_name), binary=False)

	end_time = time.time()
	print(f"\n[*] Success: Model saved as '{model_name}'")
	print(f"[*] Total Training Time: {end_time - start_time:.2f}s")
	print("-" * 40)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="FastText Training Script for TechNowNet")
	parser.add_argument('--db', type=str, required=True, help="Database name (e.g., tech, sci, mas)")
	parser.add_argument('--year_start', type=int, default=2019)
	parser.add_argument('--year_end', type=int, default=2023)
	parser.add_argument('--vector_size', type=int, default=100, help="Embedding dimension size")
	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--window', type=int, default=5)
	parser.add_argument('--sg', type=int, default=1, help="1 for Skip-gram, 0 for CBOW")
	parser.add_argument('--wdir', type=str, default='./', help="Working directory path")
	
	args = parser.parse_args()

	train(
		wdir=args.wdir,
		db=args.db,
		year_start=args.year_start,
		year_end=args.year_end,
		vector_size=args.vector_size,
		epochs=args.epochs,
		window=args.window,
		sg=args.sg
	)