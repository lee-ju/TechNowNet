import subprocess
import os

def run_script(command):
    print(f"Executing: {' '.join(command)}")
    subprocess.run(command, check=True)

if __name__ == "__main__":
    YEAR_START = 2019
    YEAR_END = 2023
    WDIR = "./"
    
    # 1. Preprocessing (Domain-specific Lemmatization)
    run_script(["python", "prep-technology.py", "--year_start", str(YEAR_START), "--year_end", str(YEAR_END), "--wdir", WDIR])
    run_script(["python", "prep-science.py", "--year_start", str(YEAR_START), "--year_end", str(YEAR_END), "--wdir", WDIR])
    run_script(["python", "prep-market-and-society.py", "--year_start", str(YEAR_START), "--year_end", str(YEAR_END), "--wdir", WDIR])

    # 2. Text corpus conversion (excluding market-and-society)
    for db in ["tech", "sci"]:
        run_script(["python", "docu.py", "--db", db, "--year_start", str(YEAR_START), "--year_end", str(YEAR_END), "--wdir", WDIR])

    # 3. FastText Learning (Domain-specific Embeddings)
    for db in ["tech", "sci", "mas"]:
        run_script(["python", "train.py", "--db", db, "--year_start", str(YEAR_START), "--year_end", str(YEAR_END), "--wdir", WDIR, "--vector_size", "100"])

    # 4. Cross-domain Alignment (created by TechNowNet)
    run_script(["python", "TechNowNet.py", "--year_start", str(YEAR_START), "--year_end", str(YEAR_END), "--wdir", WDIR, "--dim", "100"])