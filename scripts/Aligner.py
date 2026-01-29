import os
import time
import numpy as np
import pandas as pd
import argparse
from gensim.models import KeyedVectors
from scipy.linalg import orthogonal_procrustes
from sklearn.preprocessing import normalize

class alinger:
	def __init__(self, wdir, dim, year_start, year_end=2023):
		self.wdir = wdir
		self.dim = dim
		self.year_start = year_start
		self.year_end = year_end
		self.models = []
		self.vocabs = []
		self.db_names = ['sci', 'tech', 'mas']

	def load_models(self):
		print(f"[*] Loading models for dimension {self.dim}...")
		for db in self.db_names:
			path = os.path.join(self.wdir, 'model', f'{db}.{self.year_start}-{self.year_end}.lem.docu.ftx.d{self.dim}')
			print(f"  > Loading: {path}")
			try:
				model = KeyedVectors.load_word2vec_format(path, binary=False)
				model_dict = {word: model[word] for word in model.index_to_key if word is not None}
				self.models.append(model_dict)
				self.vocabs.append(set(model_dict.keys()))
			except FileNotFoundError:
				print(f"  [Error] File not found: {path}")
		return len(self.models) == len(self.db_names)

	def align_shared_terms(self):
		print("[*] Aligning shared terms...")
		shared_vocab = sorted(list(set.intersection(*self.vocabs)))
		print(f"  > Number of shared terms (p): {len(shared_vocab):,}")

		# Generate a common term matrix (Z_i) for each domain
		z_matrices = []
		for model_dict in self.models:
			z_i = np.array([model_dict[word] for word in shared_vocab])
			z_matrices.append(z_i)

		# Calculating the mean matrix (M)
		mean_matrix = np.mean(z_matrices, axis=0)

		# Calculating the optimal orthogonal matrix (W_i) using OPA
		rotation_matrices = []
		for z_i in z_matrices:
			w_i, _ = orthogonal_procrustes(z_i, mean_matrix) # W* = UV^T
			rotation_matrices.append(w_i)
		
		return shared_vocab, rotation_matrices

	def fuse_all_knowledge(self, shared_vocab, rotation_matrices):
		# Combining shared and domain-specific knowledge to create TechNowNet
		print("[*] Fusing all knowledge (Shared + Domain-specific)...")
		# Union of all domain terms (V_total)
		total_vocab = sorted(list(set.union(*self.vocabs)))
		print(f"  > Total vocabulary size (p+q): {len(total_vocab):,}")

		final_vectors = []
		start_time = time.time()

		for i, word in enumerate(total_vocab):
			aligned_vecs = []
			for idx, model_dict in enumerate(self.models):
				if word in model_dict:
					# Move to the same space by applying a rotation matrix
					aligned_vec = model_dict[word] @ rotation_matrices[idx]
					aligned_vecs.append(aligned_vec)
			
			# If it spans multiple domains, take the average
			final_vectors.append(np.mean(aligned_vecs, axis=0))

			if (i + 1) % 100000 == 0:
				print(f"  > Progress: {((i+1)/len(total_vocab))*100:.1f}%", end='\r')

		return total_vocab, np.array(final_vectors)

	def post_process(self, matrix):
		# Final vector postprocessing: mean centering and unit length normalization
		print("\n[*] Running post-processing...")
		# Mean Centering
		matrix -= np.mean(matrix, axis=0)
		# Unit Length Scaling (L2-normalized)
		return normalize(matrix)

	def save_result(self, vocab, matrix, name="TechNowNet"):
		save_path = os.path.join(self.wdir, 'model', f'{name}.d{self.dim}')
		print(f"[*] Saving final TechNowNet to: {save_path}")
		with open(save_path, 'w', encoding='utf-8') as f:
			f.write(f"{len(vocab)} {self.dim}\n")
			for word, vec in zip(vocab, matrix):
				vec_str = ' '.join(map(str, vec))
				f.write(f"{word} {vec_str}\n")

	def run(self):
		if not self.load_models(): return
		
		shared_vocab, r_matrices = self.align_shared_terms()
		total_vocab, fused_matrix = self.fuse_all_knowledge(shared_vocab, r_matrices)
		
		final_matrix = self.post_process(fused_matrix)
		self.save_result(total_vocab, final_matrix)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--wdir', type=str, default='./')
	parser.add_argument('--dim', type=int, default=100)
	parser.add_argument('--year_start', type=int, default=2019)
	parser.add_argument('--year_end', type=int, default=2023)
	args = parser.parse_args()

	TechNowNet = alinger(args.wdir, args.dim, args.year_start, args.year_end)
	TechNowNet.run()
