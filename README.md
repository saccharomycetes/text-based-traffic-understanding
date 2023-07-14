# text-based-traffic-understanding
The code and dataset of the KDD23 paper 'A Study of Situational Reasoning for Traffic Understanding' See the full paper [here](https://arxiv.org/pdf/2306.02520.pdf)


# Datasets README

## Introduction

This repository contains three datasets: `tv.jsonl`, `bdd.jsonl`, and `hdt.jsonl`. All datasets contain a collection of question and answer pairs and are structured in a JSON Lines format, where each line is a separate JSON object corresponding to a single data point or example. The datasets are suitable for tasks in natural language understanding, specifically, multiple-choice question answering and text classification.

## Dataset Structures

### TV Dataset

The `tv.jsonl` dataset has the following keys:

- **question**: A text string representing a question related to scenarios or situations typically observed in TV programs.
- **candidates**: A list of text strings representing possible answers to the question.
- **answer**: An integer that serves as an index into the list of candidates, corresponding to the correct answer.
- **class**: A character serving as a classification label for the question.
- **id**: An integer that uniquely identifies each row or question in the dataset.

### BDD Dataset

The `bdd.jsonl` dataset has a similar structure but with a few differences:

- **question**: A text string representing a question.
- **candidates**: A list of text strings representing possible answers to the question.
- **answer**: An integer that serves as an index into the list of candidates, corresponding to the correct answer.
- **class**: A more human-readable text string serving as a classification label for the question.

Please note that the `bdd.jsonl` dataset does not include an `id` field.

### HDT Dataset

The `hdt.jsonl` dataset contains question and answer pairs related to driving tests:

- **question**: A text string representing a question, typically related to driving rules or scenarios.
- **candidates**: A list of text strings representing possible answers to the question.
- **answer**: An integer that serves as an index into the list of candidates, corresponding to the correct answer.
- **state**: A text string indicating the state from which the question was sourced.
- **type**: A text string indicating the type of driving test the question pertains to (e.g., permit, motorcycle, CDL).
- **id**: An integer that uniquely identifies each row or question in the dataset.

## Usage

These datasets could be useful for training and evaluating models in natural language understanding, specifically in the areas of multiple-choice question answering and text classification. They could also be used for studying the performance of models in understanding and answering questions about TV program content, car behavior in traffic situations, and driving rules or scenarios.

Remember to split the datasets appropriately into training, validation, and test sets when using them to build machine learning models. The `id` field in the `tv.jsonl` and `hdt.jsonl` datasets may be useful for tracking individual examples, but it likely should not be used as a feature in a model.
