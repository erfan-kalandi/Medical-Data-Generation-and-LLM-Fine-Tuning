# Medical Data Generation and LLM Fine-Tuning

This project explores the capabilities of Large Language Models (LLMs) in generating and analyzing specialized medical data in Persian (Farsi). It involves three main steps:

1. **Synthetic Data Generation:** Creating realistic artificial medical data based on symptoms and diagnoses.
2. **Fine-Tuning:** Adapting a pre-trained LLaMA language model using Low-Rank Adaptation (LoRA) to better understand medical contexts.
3. **Evaluation:** Comparing model performance before and after fine-tuning using a custom medical multiple-choice question dataset.

## Features

* Loading and formatting a dataset of Persian medical symptom-diagnosis pairs.
* Fine-tuning the `unsloth/Llama-3.2-1B` model with LoRA for efficient adaptation.
* Evaluating the model on Persian medical multiple-choice questions.
* Saving and analyzing evaluation results for both base and fine-tuned models.

## Requirements

* Python 3.7+
* `transformers`
* `datasets`
* `peft`
* `torch`

Install dependencies using:

```bash
pip install datasets transformers peft torch
```

## Usage

1. **Prepare the dataset:**

   * `symptoms_diagnoses_100.json` containing medical cases.
   * `qa.json` containing multiple-choice questions with answers.

2. **Run the notebook/script:**
   The code:

   * Loads and formats the medical dataset.
   * Loads the pre-trained LLaMA model and tokenizer.
   * Evaluates base model accuracy on the question dataset.
   * Applies LoRA fine-tuning on the synthetic data.
   * Trains the model with specified hyperparameters.
   * Evaluates the fine-tuned model.
   * Saves evaluation results and the fine-tuned model.


## Results

* Model accuracy before fine-tuning is saved in `evaluation_base_model.json`.
* Model accuracy after fine-tuning is saved in `evaluation_finetuned_model.json`.
* The fine-tuned model and tokenizer are saved in the `./fine-tuned-model` directory.

## Notes

* The project is designed for Persian medical NLP applications.
* Uses causal language modeling tailored for question answering.
* LoRA enables efficient fine-tuning by training a small subset of model parameters.
