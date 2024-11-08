# AIMixDetect: detect mixed authorship of a language model (LM) and humans

---
contributors:
  - Alon Kipnis
  - Idan Kashtan
---

## Overview

This replication package is designed to guide you through the process of replicating the results presented in the paper:
- Kashtan, I., & Kipnis, A. (2024). "An Information-Theoretic  Approach for Detecting Edits in AI-Generated Text." Harvard Data Science Review, (Special Issue 5).

This paper uses a new dataset generated using GPT-3.5-turbo (ChatGPT) and is organized into five distinct datasets, which are located in the `dataset` folder and HuggingFace dataset. To facilitate the reading and parsing of these datasets, a script named `parse_article.py` is provided.

The analysis conducted in the paper is based on the construction of specific analysis files for three statistical methods: Higher Criticism (HC) of logperpelxity P-values, the minimal P-value (minP) of such P-values, and a logistic regression (LoR) classifier applied to the embedding of full articles. The code included in this package allows you to replicate the results in the paper. Specifically, three main scripts—`simulate_replication_HC.py`, `simulate_replication_LoR.py`, and `simulate_replication_HC_power_analysis.py`—are used to run the core analyses discussed in the paper.

## Data Availability and Provenance Statements

### Statement about Rights

- [x] I certify that the author(s) of the manuscript have legitimate access to and permission to use the data used in this manuscript. 
- [x] I certify that the author(s) of the manuscript have documented permission to redistribute/publish the data contained within this replication package. Appropriate permission are documented in the [LICENSE.txt](LICENSE.txt) file.

### Summary of Availability

- [x] All data **are** publicly available.
- [ ] Some data **cannot be made** publicly available.
- [ ] **No data can be made** publicly available.

### Details on each Data Source

| Data.Name  | Data.Files | Location |
| -- | -- | -- | 
| “Geographical landmarks articles” | [AIMixDetect](https://huggingface.co/datasets/Kashtan/AIMixDetect) | Hugging Face |
| “Historical figures articles” | [AIMixDetect](https://huggingface.co/datasets/Kashtan/AIMixDetect) | Hugging Face |
| “Nature articles” | [AIMixDetect](https://huggingface.co/datasets/Kashtan/AIMixDetect) | Hugging Face |
| “Video games articles” | [AIMixDetect](https://huggingface.co/datasets/Kashtan/AIMixDetect) | Hugging Face |
| “Wars articles” | [AIMixDetect](https://huggingface.co/datasets/Kashtan/AIMixDetect) | Hugging Face |
| “Abstract Dataset” | [AIMixDetectPublicData](https://huggingface.co/datasets/Kashtan/AIMixDetectPublicData) | Hugging Face |
| “News Dataset” | [AIMixDetectPublicData](https://huggingface.co/datasets/Kashtan/AIMixDetectPublicData) | Hugging Face |
| “Wiki Dataset” | [AIMixDetectPublicData](https://huggingface.co/datasets/Kashtan/AIMixDetectPublicData) | Hugging Face |

The data used to support the findings of this study have been deposited in HuggingFace. The authors created the data which are available under a Creative Commons Non-commercial license. For each data source, we provide three types of edit ratios: 0.05, 0.1, and 0.15. You can find a parser and an explanation of how to read the data yourself in the provided link. Additionally, we used three publicly available Huggingface datasets to conduct our power analysis.

## Computational requirements

### Software Requirements

- [x] The replication package contains one or more programs to install all dependencies and set up the necessary directory structure.

- Python 3.10.11
  - `pandas` 2.0.0
  - `numpy` 1.24.1
  - `tqdm` 4.65.0
  - `argparse`
  - `pyyaml`
  - `scikit-learn` 1.2.2
  - `transformers` 4.36.2
  - `torch` 1.13.1
  - `multiple-hypothesis-testing` 0.1.12
  - `logging`
  - `scipy` 1.10.1
  - the file "`requirements.txt`" lists these dependencies, please run "`pip install -r requirements.txt`" as the first step.

### Controlled Randomness

- [x] Random seed is set at line 48 of program `simulate_replication_LoR.py`
- [ ] No Pseudo random generator is used in the analysis described here.

### Memory, Runtime, Storage Requirements

Approximate time needed to reproduce the analyses on a standard 2024 desktop machine:

- [ ] <10 minutes
- [x] 10-60 minutes
- [ ] 1-2 hours
- [ ] 2-8 hours
- [ ] 8-24 hours
- [ ] 1-3 days
- [ ] 3-14 days
- [ ] > 14 days

Approximate storage space needed:

- [ ] < 25 MBytes
- [x] 25 MB - 250 MB
- [ ] 250 MB - 2 GB
- [ ] 2 GB - 25 GB
- [ ] 25 GB - 250 GB
- [ ] > 250 GB

- [ ] Not feasible to run on a desktop machine, as described below.

#### Details

The code was last run on a **Windows 10 laptop with an Intel Core i7-8550U CPU @ 1.80GHz (1.99 GHz) and 16GB of RAM**. 

## Description of programs/code

- Programs in `src` are the util functions and classes for our method.
- Program `simulate_replication_HC.py` will simulate the results for HC and minP over the provided dataset.
- Program `simulate_replication_LoR.py` will simulate the results for Logistic regression over the provided dataset.
- Program `simulate_replication_HC_power_analysis.py` will simulate the results for the power analysis over Huggingface data.
- `simulate_replication_conf.yml` a configuration file.
- Zip file `cache_files/cache_files.zip` contains the cache files for faster simulation, pls unzip it before using it.

## Instructions to Replicators

- Edit `simulate_replication_conf.yml` to adjust the parameters and datasets.
- Unzip the `cache_files/cache_files.zip` to use cache and make the simulation faster.
- Run `simulate_replication_HC.py` to simulate the results for HC and minP over the provided dataset.
- Run `simulate_replication_LoR.py` to simulate the results for Logistic regression over the provided dataset.
- Run `simulate_replication_HC_power_analysis.py` to simulate the results for the power analysis over Huggingface data.

### Details

- `simulate_replication_HC.py` will create a csv file to the desired destination containing the results
- `simulate_replication_LoR.py` will create a csv file to the desired destination containing the results
- `simulate_replication_HC_power_analysis.py` will create a csv file to the desired destination containing the results

## List of tables and programs

The provided code reproduces:

- [ ] All numbers provided in text in the paper
- [ ] All tables and figures in the paper
- [x] Selected tables and figures in the paper, as explained and justified below.

| Figure/Table #    | Program                                   | Line Number | Output file                |
|-------------------|-------------------------------------------|-------------|----------------------------|
| Figure 6          | simulate_replication_HC_power_analysis.py |             | results_power_analysis.csv |
| Figure 7          | simulate_replication_HC.py                |             | results.csv                |
| Figure 8          | simulate_replication_HC.py                |             | results.csv                |
| Figure 8          | simulate_replication_LoR.py               |             | results_LoR.csv            |

## Citation Information
```
@article{Kashtan2024Information,
  author = {Kashtan, Idan and Kipnis, Alon},
  journal = {Harvard Data Science Review},
  number = {Special Issue 5},
  year = {2024},
  month = {aug 15},
  note = {https://hdsr.mitpress.mit.edu/pub/f90vid3h},
  publisher = {The MIT Press},
  title = {
    {An} {Information}-{Theoretic}
    {Approach} for {Detecting} {Edits} in {AI}-{Generated} {Text} },
  volume = { },
}
```
