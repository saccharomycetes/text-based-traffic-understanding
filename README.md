# text-based-traffic-understanding
The dataset of the KDD23 paper 'A Study of Situational Reasoning for Traffic Understanding' See the full paper [here](https://arxiv.org/pdf/2306.02520.pdf)


# Datasets README

## Introduction

This repository contains four datasets: `tv.jsonl`, `bdd.jsonl`, `hdt.jsonl` and `manuals.jsonl`. All datasets are structured in a JSON Lines format, where each line is a separate JSON object corresponding to a single data point or example.

## Dataset Structures

### Complex-TV-QA Dataset

The `tv.jsonl` dataset has the following keys:

- **description**: A detailed description of a traffic video
- **question**: A question that is related to the video
- **candidates**: A list of text strings representing possible answers to the question
- **answer**: An integer that serves as an index into the list of candidates, corresponding to the correct answer
- **class**: The reasoning types defined in the original [Traffic-QA](https://arxiv.org/pdf/2103.15538.pdf) paper
- **video_file**: The original video file from the Traffic-QA dataset, you may request the download from [here](https://github.com/SUTDCV/SUTD-TrafficQA)
- **id**: An integer that uniquely identifies each question in the dataset

### BDD-QA Dataset

The `bdd.jsonl` dataset has the following keys:

- **question**: A text string which briefly describe a traffic senario and a follow up question
- **candidates**: A list of text strings representing possible answers to the question
- **answer**: An integer that serves as an index into the list of candidates, corresponding to the correct answer
- **class**: The action class that this question is related to, which is defined in the paper
- **id**: An integer that uniquely identifies each question in the dataset

### HDT-QA Dataset

The `hdt.jsonl` dataset has the following keys:

- **question**: A text string representing a question, typically related to driving rules or scenarios.
- **candidates**: A list of text strings representing possible answers to the question.
- **answer**: An integer that serves as an index into the list of candidates, corresponding to the correct answer.
- **state**: A text string indicating the state from which the question was sourced.
- **type**: A text string indicating the type of driving test the question pertains to (e.g., permit, motorcycle, CDL).
- **id**: An integer that uniquely identifies each row or question in the dataset.

The `manuals.jsonl` dataset is a large collection of driving manuals from 51 states of the US, which is oringally crawled from [DMV TEST PRO](https://www.dmv-test-pro.com/), which contains the following keys:

- **text**: A paragraph from the original DMV driving manual
- **state**: The state of the source of this driving manual
- **domain**: The knowledge domain of this driving manual, could be one of: `CDL`, `permit`, `motorcycle`


## Usage

The Complex-TV-QA dataset, to our knowledge, is the inaugural resource that provides human-annotated, detailed video captions within traffic scenarios, alongside complex reasoning questions. This novel dataset not only stands as a vital tool for evaluating language models in real-world video-QA and video-reasoning research, but also offers valuable insights for the development and understanding of multi-modal video reasoning models and related works.

BDD-QA is distinguished by its encompassing range of traffic actions, crafted to rigorously evaluate a model's decision-making abilities in traffcu senario. This makes it a potent tool for high-level decision-making research within traffic contexts, including autonomous driving developments.

HDT-QA, coupled with driving manuals, offers an extensive compendium of driving instructions and driving knowledge tests across all 51 states of the US. This resource is beneficial for assessing the incorporation and impact of traffic knowledge within intelligent driving systems, marking a crucial stride towards more advanced, informed, and safe autonomous driving technology.

## Cite 
```
@article{zhang2023study,
  title={A Study of Situational Reasoning for Traffic Understanding},
  author={Zhang, Jiarui and Ilievski, Filip and Ma, Kaixin and Kollaa, Aravinda and Francis, Jonathan and Oltramari, Alessandro},
  journal={arXiv preprint arXiv:2306.02520},
  year={2023}
}
```

## Contact

-   `jrzhang [AT] isi.edu`