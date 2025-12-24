
## Assignment Checklist

### Setup & Environment
- [ ] Run `bash setup.sh` successfully
- [ ] Verify model weights downloaded (stories42M.pt)

### Core Implementation
- [ ] **llama.py**
- [ ] **rope.py**
- [ ] **optimizer.py**
- [ ] **classifier.py**
- [ ] **lora.py**

### Testing & Validation
- [ ] Pass `python sanity_check.py`
- [ ] Pass `python optimizer_test.py` 
- [ ] Pass `python rope_test.py` 
- [ ] Generate coherent text with `python run_llama.py --option generate`
- [ ] Complete SST zero-shot prompting
- [ ] Complete CFIMDB zero-shot prompting  
- [ ] Complete SST fine-tuning
- [ ] Complete CFIMDB fine-tuning
- [ ] Complete SST LoRA fine-tuning
- [ ] Complete CFIMDB LoRA fine-tuning

### Advanced Features (Optional - 3 points)
- [ ] Other advanced techniques (see advanced options above)
- [ ] Write 1-2 page report documenting improvements

### Submission
- [ ] Generate all required output files
- [ ] Any model weights (if needed for advanced implementation) need to be hosted on your own drive with link shared in the report
- [ ] Submission procedure is the same as in HW1

