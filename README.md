# Min-Llama Assignment 

<div align="center">
  <img src="mini_llama.jpeg" alt="ANLP mini-llama" width="200">
</div>

Based on https://github.com/cmu-l3/anlp-fall2025-hw1/tree/main

## Overview

In this assignment, you will implement important components of the Llama2 model to better understand its architecture. You will then perform sentence classification on the SST and CFIMDB datasets with this model.

## Assignment Details

### Your Task

You are responsible for implementing core components of Llama2 in the following files:
- `llama.py` - Main model architecture
- `classifier.py` - Classification head
- `optimizer.py` - AdamW optimizer  
- `rope.py` - Rotary position embeddings
- `lora.py` - LoRA implementation

You will work with `stories42M.pt`, an 8-layer, 42M parameter language model pretrained on the [TinyStories](https://arxiv.org/abs/2305.07759) dataset (machine-generated children's stories). This model is small enough to train without a GPU, though using Colab or a personal GPU is recommended for faster iteration.

### Testing Your Implementation

Once you have implemented the components, you will test your model in four settings:

1. **Text Generation**: Generate completions starting with: *"I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"*. You should see coherent, grammatical English (though content may be absurd due to the children's stories training data).

2. **Zero-shot Prompting**: Perform prompt-based sentiment analysis on SST-5 and CFIMDB datasets. This will give poor results (roughly random performance).

3. **Fine-tuning**: Perform task-specific fine-tuning with a classification head. This will give much stronger results.
4. **LoRA Fine-tuning**: Perform task-specific LoRA fine-tuning with a classification head. 

5. **Advanced Implementation**: Implement something new on top of the base requirements for potential extra credit.

### Important Notes

- Follow `setup.sh` to properly set up the environment and install dependencies
- See [structure.md](./structure.md) for detailed code structure descriptions
- Use only libraries installed by `setup.sh` - no external libraries (e.g., `transformers`) allowed
- The `data/cfimdb-test.txt` file contains placeholder labels (-1), so test accuracies may appear low
- Ensure reproducibility using the provided commands
- Do not change existing command options or add new required parameters
- Refer to [checklist.md](./checklist.md) for assignment requirements

## Reference Commands and Expected Results

### Text Generation
```bash
python run_llama.py --option generate
```
You should see continuations of the sentence `I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is...`. We will generate two continuations - one with temperature 0.0 (which should have a reasonably coherent, if unusual, completion) and one with temperature 1.0 (which introduces randomness for more creative and diverse outputs, though potentially less coherent).


### Zero-Shot Prompting

**SST Dataset:**
```bash
python run_llama.py --option prompt --batch_size 10 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt [--use_gpu]
```
- Dev Accuracy: 0.237 (0.000)
- Test Accuracy: 0.250 (0.000)

**CFIMDB Dataset:**
```bash
python run_llama.py --option prompt --batch_size 10 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt [--use_gpu]
```
- Dev Accuracy: 0.490 (0.000)
- Test Accuracy: 0.109 (0.000)

### Classification Fine-tuning

**SST Dataset:**
```bash
python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt [--use_gpu]
```
- Dev Accuracy: 0.411 (0.025)
- Test Accuracy: 0.399 (0.023)

**CFIMDB Dataset:**
```bash
python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt [--use_gpu]
```
- Dev Accuracy: 0.833 (0.060)
- Test Accuracy: 0.473 (0.198)

### LoRA Fine-tuning

**SST Dataset:**
```bash
python run_llama.py --option lora --epochs 5 --lr 2e-5 --batch_size 80 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-lora-output.txt --test_out sst-test-lora-output.txt --lora_rank 4 --lora_alpha 1.0 [--use_gpu]
```
- Dev Accuracy: 0.275 (0.024)
- Test Accuracy: 0.269 (0.020)

**CFIMDB Dataset:**
```bash
python run_llama.py --option lora --epochs 5 --lr 2e-5 --batch_size 10 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-lora-output.txt --test_out cfimdb-test-lora-output.txt --lora_rank 4 --lora_alpha 1.0 [--use_gpu]
```
- Dev Accuracy: 0.510 (0.051)
- Test Accuracy: 0.506 (0.239)

*Note: Mean reference accuracies over 10 random seeds with standard deviations in brackets.*

## Submission Requirements

### Submission Guideline

**Code:**
Submission procedure is the same as in HW1, any model weights (if needed for advanced implementation) need to be hosted on your own drive with link shared in the report.


**Report (optional; mandatory for 10 consideration):** Add pdf file, named report.pdf, if (1) you've implemented something else on top of the requirements and further improved accuracy for possible extra points (see "Grading" below), and/or (2) if your best results are with some hyperparameters other than the default, and you want to specify how we should run your code. If you're doing (1), we expect your report should be 1-2 pages, but no more than 3 pages. If you're doing (2), the report can be very brief.



### Grading

* **10**: (Advanced implementation) You must implement something additional on top of the requirements for 7 and achieve significant accuracy improvements or demonstrate exceptional creativity. This improvement can be in either the zero-shot setting (no task-specific fine-tuning required) or in the fine-tuning setting (improving over our current fine-tuning implementation). Ensure that your implementation is still executable using the specified commands. Please write down what you implemented and the experiments you performed in the report, along with clear instructions for how to run them. **You will not be eligible for an 10 without the report**. You are welcome to provide additional materials such as commands to run your code in a script and training logs.
   * Perform [continued pre-training](https://arxiv.org/abs/2004.10964) using the language modeling objective to do domain adaptation
   * Enable zero-shot prompting using a more principled inference algorithm than our current implementation. For example, we did not include an attention mask despite right-padding all inputs (to enable batch prediction); this could be improved.
   * Explore other [decoding mechanisms](https://arxiv.org/abs/2402.06925). You must experiment with at least two different decoding techniques beyond the baseline and demonstrate measurable improvement over the baseline model's performance.
   * Perform [prompt-based fine-tuning](https://arxiv.org/abs/2109.01247)
   * Add [regularization](https://arxiv.org/abs/1909.11299) to our fine-tuning process
   * Try other variants of positional encodings (or potential [replacements](https://arxiv.org/pdf/2108.12409)) that consistently improve the current model's performance
   * Try parameter-efficient fine-tuning (see Section 2.2 [here](https://arxiv.org/abs/2110.04366) for an overview)
   * Try alternative fine-tuning algorithms e.g. [SMART](https://www.aclweb.org/anthology/2020.acl-main.197) or [WiSE-FT](https://arxiv.org/abs/2109.01903)
   * Add other model components on top of the model
   * Implement alternative attention mechanisms such as sparse attention, [Differential Transformer](https://arxiv.org/pdf/2410.05258), or other attention variants that improve model performance compared to standard attention.

* **7**: You implement all the missing pieces and the original `classifier.py` and  `lora.py`  with `--option prompt`, `--option finetune` and `--option lora` code such that coherent text (i.e. mostly grammatically well-formed) can be generated and the model achieves comparable accuracy to our reference implementation.

* **6**: You implement all the missing pieces and the original `classifier.py` and  `lora.py`  with `--option prompt`, `--option finetune` and `--option lora` code, but coherent text is not generated (i.e. generated text is not well-formed English) or accuracy is not comparable to the reference.

* **5**: All missing pieces are implemented and pass tests in `sanity_check.py` (llama implementation) and `optimizer_test.py` (optimizer implementation).

* **4 or below**: Some parts of the missing pieces are not implemented.


### Acknowledgement

This code is based on llama2.c by Andrej Karpathy. Parts of the code are also from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).