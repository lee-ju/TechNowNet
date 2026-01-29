import os
import time
import argparse
import re
from multiprocessing import Pool
from collections import defaultdict

NON_ENG_PATTERN = re.compile(r'[\uAC00-\uD7AF\u3040-\u30FF\u4E00-\u9FFF]')

class DocumentFormatter:
	def __init__(self, train_start, train_end):
		self.train_start = train_start
		self.train_end = train_end

	def process_line(self, line):
		parts = line.strip().split('<SEP>')
		if not parts:
			return False, None, ""

		date_str = parts[0]
		try:
			year = int(date_str[:4])
		except ValueError:
			return False, date_str, ""

		if self.train_start <= year <= self.train_end:
			content_parts = parts[1:]
			
			if len(content_parts) >= 2:
				texts = [t for t in content_parts[0].split('\t') if not NON_ENG_PATTERN.search(t)]
				
				if len(content_parts[1]) >= 1:
					auth_kwds = [a for a in content_parts[1].split('\t') if not NON_ENG_PATTERN.search(a)]
					merged_tokens = ' '.join(texts) + ' ' + ' '.join(auth_kwds)
				else:
					merged_tokens = ' '.join(texts)
				return True, date_str, merged_tokens.strip()

			elif len(content_parts) == 1:
				tokens = [t for t in content_parts[0].split('\t') if not NON_ENG_PATTERN.search(t)]
				return True, date_str, ' '.join(tokens).strip()
		
		return False, date_str, ""

def formatter():
	parser = argparse.ArgumentParser(description="Create FastText training corpus (.docu) for TechNowNet")
	parser.add_argument('--db', choices=['tech', 'sci', 'mas'], default='patstat', help='Target database name')
	parser.add_argument('--year_start', type=int, default=2019, help='Start year for training data')
	parser.add_argument('--year_end', type=int, default=2023, help='End year for training data')
	parser.add_argument('--recent5', type=str, choices=['Y', 'N'], default='Y', help='Save additional file for recent 5 years (Y/N)')
	parser.add_argument('--n_cpus', type=int, default=8, help='Number of CPUs for parallel processing')
	parser.add_argument('--wdir', type=str, default='./dataset/', help='Working directory for datasets')

	args = parser.parse_args()
	
	year_recent5 = args.year_end - 5 + 1
	
	input_filename = f"{args.db}.{args.year_start}-{args.year_end}.lem"
	input_path = os.path.join(args.wdir, input_filename)
	
	if not os.path.exists(input_path):
		print(f"[Error] File not found: {input_path}")
		return

	print(f"--- Processing: {args.db} ({args.year_start}-{args.year_end}) ---")
	
	with open(input_path, 'r', encoding='utf-8') as f:
		lines = f.readlines()

	formatter = DocumentFormatter(args.year_start, args.year_end)
	
	f_total = open(input_path + '.docu', 'w', encoding='utf-8')
	f_recent = None
	if args.recent5 == 'Y':
		recent_path = os.path.join(args.wdir, f"{args.db}.{year_recent5}-{args.year_end}.lem.docu")
		f_recent = open(recent_path, 'w', encoding='utf-8')

	year_stats = defaultdict(int)
	total_count = len(lines)
	start_time = time.time()

	print(f"--- Starting Document Transformation (Total: {total_count:,} lines) ---")
	
	with Pool(args.n_cpus) as pool:
		for i, (success, date, tokens) in enumerate(pool.imap(formatter.process_line, lines)):
			if success and tokens:
				f_total.write(tokens + '\n')
				year_stats[date[:4]] += 1
				
				if f_recent and int(date[:4]) >= year_recent5:
					f_recent.write(tokens + '\n')
			
			if (i + 1) % (max(1, total_count // 100)) == 0:
				elapsed = time.time() - start_time
				print(f" > Progress: {((i+1)/total_count)*100:.1f}% | Elapsed: {elapsed:.1f}s", end='\r')

	f_total.close()
	if f_recent:
		f_recent.close()

	print(f"Total processed sentences: {sum(year_stats.values()):,}")
	for yr in sorted(year_stats.keys()):
		print(f"  > Year {yr}: {year_stats[yr]:,} sentences")

if __name__ == '__main__':
	formatter()
