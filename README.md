# BiGTex
# Integrating Structural and Semantic Signals in Text-Attributed Graphs with BiGTex
[![arXiv](https://arxiv.org/abs/2504.12474)



## First, Download TAG datasets



| Dataset | Description |
| ----- |  ---- |
| ogbn-arxiv  | The [OGB](https://ogb.stanford.edu/docs/nodeprop/) provides the mapping from MAG paper IDs into the raw texts of titles and abstracts. <br/>Download the dataset [here](https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz), unzip and move it to `dataset/ogbn_arxiv_orig`.|
| ogbn-products (subset) |  The dataset is located under `dataset/ogbn_products_orig`.|
| arxiv_2023 |  Download the dataset [here](https://drive.google.com/file/d/1-s1Hf_2koa1DYp_TQvYetAaivK9YDerv/view?usp=sharing), unzip and move it to `dataset/arxiv_2023_orig`.|
|Cora| Download the dataset [here](https://drive.google.com/file/d/1hxE0OPR7VLEHesr48WisynuoNMhXJbpl/view?usp=share_link), unzip and move it to `dataset/cora_orig`.|
PubMed | Download the dataset [here](https://drive.google.com/file/d/1sYZX-jP6H8OkopVa9cp8-KXdEti5ki_W/view?usp=sharing), unzip and move it to `dataset/PubMed_orig`.|

## Training and then save embeddings for BiGTex and ogbn-arxiv

```
python main.py 'arxiv' 'BiGTex'
```
you can run for other dataset: 'pubmed', 'products', 'arxiv_2023'
or other models: 'MLP', 'GCN', 'GAT', 'SAGE'

## BiGTex embeddings
You can download the generated embeddings by BiGTex [here](https://drive.google.com/drive/folders/1nF8NDGObIqU0kCkzVaisWooGEQlcNSIN?usp=sharing).
after that move them to `embeddings`, so you can run more experiments like link prediction or clusstering using them.
