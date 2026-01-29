# TechNowNet: A Contemporary Multi-Domain Knowledge Semantic Network

TechNowNet is a multi-domain semantic network designed to integrate science, technology, and market-and-society knowledge.
It employs cross-domain embedding fusion to align knowledge from heterogeneous sources (Scopus, PATSTAT, NewsCatcher) into a single semantic space.

---

## Project Structure

```text
TechNowNet/
├── data/                            # Original source data (Scopus, PATSTAT, NewsCatcher)
├── dataset/                         # Preprocessed results (.lem) and training corpora (.docu)
├── model/                           # Trained FastText models and integrated vectors
├── scripts/                         # Module source codes
│   ├── prep-technology.py           # Patent data preprocessing
│   ├── prep-science.py              # Scopus data preprocessing
│   ├── prep-market-and-society.py   # Market-and-society data preprocessing
│   ├── utils.py                     # Utility for preprocessing
│   ├── docu.py                      # Training corpus (.docu) generation
│   ├── train.py                     # FastText domain-specific training
│   └── Aligner.py                   # OPA-based cross-domain fusion (TechNowNet)
├── main.py                          # Integrated execution script
├── setup.py                         # Setup for TechNowNet
├── requirements.txt                 # Required library list
└── README.md                        # Project documentation
```

---

## Execution Pipeline

TechNowNet processes knowledge through a circular structure of generation, application, and cognition.

### 1. Preprocessing (`prep-*.py`)
Extracts and cleans terms from specific domains for the period 2019-2023
* Performs tokenization, N-gram generation, and lemmatization.
* Extracts shared terms (appearing in all domains) and domain-specific terms.

### 2. Training Corpus Generation (`docu.py`)
Converts `.lem` files into `.docu` format suitable for large-scale embedding training.
* Filters out non-English characters to ensure linguistic consistency.

### 3. Domain-specific Embedding (`train.py`)
Trains independent **FastText** models for each domain to preserve morphological and subword information.
* Independent training maximizes the retention of domain-specific uniqueness.

### 4. Cross-domain Fusion (`Aligner.py`)
Uses an **extended Orthogonal Procrustes Analysis (OPA)** to align multiple embedding spaces.
* **Shared Knowledge Fusion**: Aligns shared term matrices towards a mean matrix.
* **Domain-specific Alignment**: Positions unique terms within the fused space while preserving their specialized features.

---

## Key Achievements
* **Scalability**: Achieved approximately 4.4 times greater term coverage (17.7 million terms) than existing methods.
* **Quality**: Demonstrated a 7-percentage-point higher performance (**SOTA**) in Technology-Term Relevance (TTR) tasks.
* **Contemporaneity**: Captures the latest trends in AI, quantum computing, and biotechnology.

---

## Requirements
* Python 3.8+
* Gensim 4.0+
* NLTK
* Scipy & Scikit-learn

## Citation (BibTeX)
If you use `TechNowNet` in your research, please cite [TechNowNet: A Contemporary Multi-Domain Knowledge Semantic Network through Cross-Domain Embedding Fusion](https://github.com/lee-ju/TechNowNet).

```text
@inproceedings{Lee2026TechNowNet,
  title={TechNowNet: A Contemporary Multi-Domain Knowledge Semantic Network through Cross-Domain Embedding Fusion},
  author={Lee, Juhyun and Lim, Jitaek and Yang, Heyoung},
  affiliation={Korea Institute of Science and Technology Information},
  year={2026+},
  journal={Under Review},
  publisher={...},
  url={https://github.com/lee-ju/TechNowNet},
}
```
