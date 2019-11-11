# AttrRW_Matrix_Factorization

Source code for [Attributed Random Walk as Matrix Factorization](https://grlearning.github.io/papers/95.pdf).

Authors: [Lei Chen](https://leichen2018.github.io) (NYU), Shunwang Gong (ICL), [Joan Bruna](https://cims.nyu.edu/~bruna/) (NYU), [Michael Bronstein](https://www.imperial.ac.uk/people/m.bronstein) (Twitter/ICL/USI).

[Graph Representation Learning Workshop NeurIPS 2019](https://grlearning.github.io).

## Acknowledgement

* Code of GraphRNA is cloned from [https://github.com/xhuang31/GraphRNA_KDD19](https://github.com/xhuang31/GraphRNA_KDD19).

* We would like to thank Zhengdao Chen (NYU) for suggestions and proofreading.

## Environment

PyTorch, scipy, sklearn, numpy

## Directory Initialization

```
mkdir data
mkdir results
mkdir save_model
```

## Download data

Download `BlogCatalog.mat` and `Flickr.mat` from `https://github.com/xhuang31/LANE`. Place them under `data/`.

## Running Scripts

* Ours 1

```
python main.py --model ATTR_RW_MF --dataset blogcatalog --gpu 0 --proportion 0.10 --seed 0 --output_file attr_blog_10_
python main.py --model ATTR_RW_MF --dataset blogcatalog --gpu 0 --proportion 0.10 --seed 0 --saved --output_file attr_blog_10_
```

* Ours 2

```
python main.py --model ATTR_RW_MF --dataset blogcatalog --gpu 0 --proportion 0.10 --seed 0 --output_file attr_blog_10_
python main.py --model ATTR_RW_MF --dataset blogcatalog --gpu 0 --proportion 0.10 --seed 0 --saved --output_file attr_blog_10_
```

* Ours 3

```
python main.py --model ATTR_RW_MF_3 --dataset blogcatalog --gpu 0 --proportion 0.10 --seed 0 --output_file attr_blog_10_
python main.py --model ATTR_RW_MF_3 --dataset blogcatalog --gpu 0 --proportion 0.10 --seed 0 --saved --output_file attr_blog_10_
```

* AttrRW

```
python main.py --model GRAPHRNA_RW --dataset blogcatalog --gpu 0 --proportion 0.10 --seed 0 --saved --output_file rna_blog_10_
```

* AttrRW+RNN

```
python main.py --model GRAPHRNA_RW_FULL --dataset blogcatalog --gpu 0 --proportion 0.10 --seed 0 --saved --output_file rna_blog_10_
```

* GCN

```
python main.py --model GCN --dataset blogcatalog --gpu 0 --proportion 0.1 --seed 0 --output_file gcn_blog_10_
```

* GFNN

```
python main.py --model GFNN --dataset blogcatalog --gpu 0 --proportion 0.1 --seed 0 --output_file gfnn_blog_10_
```

* NetMF

```
python main.py --model NetMF --dataset blogcatalog --gpu 0 --proportion 0.1 --seed 0 --output_file netmf_blog_10_
```

## Postprocess Results

Note that previous scripts store results for specific seeds, with a prefix in `--output_file`.

```
python average.py --output-folder results --output-file attr_blog_10_
```
