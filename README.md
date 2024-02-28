# SMGF: Spectrum-guided Multi-view Graph Fusion

This repository contains the implementation of **SMGF** algorithm.

## Prerequisites

Install dependencies by `pip install -r requirements.txt`

Unzip the content of "data/data.zip" into "data" folder. 

## Usage

Available datasets: ACM, DBLP, IMDB, Freebase, RM, Yelp, Amazon-computers, Amazon-photos, MAG-phy, MAG-eng.

### Embedding

```bash
python SMGF.py --dataset dataset_name --embedding
```

### Clustering

```bash
python SMGF.py --dataset dataset_name
```

