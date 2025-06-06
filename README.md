# Edge-Blockchain-Nexus
This repository contains the code and data for the systematic literature review of papers with blockchain and edge computing nexus.

## Methodology and Data Encoding
See `SLR-Methodology-Coding-v0.pdf` for the methodology of the systematic literature review and the data encoding.

## Paper Removal
Run `PaperRemoval.ipynb` to merge the papers from scopus, web of science, and ScienceDirect.
Then, remove the papers as the following criteria:
1. Duplicate;
2. Non-English
3. Predatory journals or conferences
4. Small pages (< 5)
5. Outdated (year < 2015)

Finally, it outputs the `final_output_file.csv` that lists the remaining papers. The dataset and output files are listed in `output_audit/`.

## Descriptive Analysis
The `descriptive_analysis/` folder contains code to derive the descriptive statistics of the reviewed papers.
The `Encoding.docx` file lists the encoding and the name of each attribute in the dataset.
Run `data_analysis.ipynb` to get statistical data and plots of the dataset. Code block 1 defines necessary functions and preprocessing steps. Code block 2-13 reads the dataset and outputs data used as input to generate graphs on rawgraphs.io. Code block 14-24 generates graphs and saves them in the `descriptive_analysis/output/` folder.

## Analytics using Machine Learning
It includes the data analysis of reviewed papers using clustering, multiple correspondence analysis (MCA), feature importance, correlation and classficiation. It lists the methodology coding and output results in `Analytics_ML/`:
1. `DimensionToBurtMatrix.py` transforms the paper data in `ReviewedRecords.xlsx` to binary data, where **1** denotes the paper studies the attribute of study dimension (e.g., PROB_TR); **0** denotes the paper does not have. It outputs the file `output/binary_matrix.csv`.
2. `AttributeReduction.py` reduces the unimportanmt attributes of each paper data, whose binary values is 0, and evaluate the results using clustering and silhouette scores, shown in `output/Clustering_feature_reduction/`.
3. `MCAbasic.py` outputs eigenvalues and contribution of attributes using MCA given the attribute-reduced binary matrix. It outputs in `output/MCA/`.
4. `MCAwithClustering.py` finds the top important attributes for MCA latent dimensions, and find centroid points using clustering. It outputs in `output/MCA/`.
5. `Classification.py` calculates correlation and feature importance using multi-tree classfication method. It outputs in `output/Classifier/`.