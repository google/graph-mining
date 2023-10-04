# The Graph Mining Library

This project includes some tools by the [Google Graph Mining team](https://research.google/teams/graph-mining/), namely in-memory
clustering. Our tools can be used for solving data mining and machine learning
problems that either inherently have a graph structure or can be formalized as
graph problems. For more information, see our [NeurIPS'20 workshop](https://gm-neurips-2020.github.io/).

Among others, this repository contains shared memory parallel clustering algorithms which scale to graphs with tens of billions of edges and are based on the following research papers:

* [Hierarchical Agglomerative Graph Clustering in Poly-Logarithmic Depth](https://papers.nips.cc/paper_files/paper/2022/hash/909de96145d97514b143dfde03e6cd2b-Abstract-Conference.html), Laxman Dhulipala, David Eisenstat, Jakub Lacki, Vahab Mirrokni, Jessica Shi, NeurIPS'22. See https://github.com/google/graph-mining/tree/main/in_memory/clustering/hac

* [Scalable community detection via parallel correlation clustering](https://dl.acm.org/doi/abs/10.14778/3476249.3476282), Jessica Shi, Laxman Dhulipala, David Eisenstat, Jakub Łącki, Vahab Mirrokni, VLDB'21. See https://github.com/google/graph-mining/tree/main/in_memory/clustering/correlation

* [Affinity Clustering: Hierarchical Clustering at Scale](https://papers.nips.cc/paper_files/paper/2017/hash/2e1b24a664f5e9c18f407b2f9c73e821-Abstract.html), Mohammadhossein Bateni, Soheil Behnezhad, Mahsa Derakhshan, MohammadTaghi Hajiaghayi, Raimondas Kiveris, Silvio Lattanzi, Vahab Mirrokni, NeurIPS'17 (the paper describes a MapReduce algorithm). See https://github.com/google/graph-mining/tree/main/in_memory/clustering/affinity

* [Distributed Balanced Partitioning via Linear Embedding](https://dl.acm.org/doi/10.1145/2835776.2835829), Kevin Aydin, MohammadHossein Bateni, Vahab Mirrokni, WSDM'16 (the paper describes a MapReduce algorithm). See https://github.com/google/graph-mining/tree/main/in_memory/clustering/parline

For questions/comments, please create an issue on this repository.

## Quickstart

1. Install [Bazel](https://bazel.build/)
2. Run the example: `bazel run //examples:quickstart`
