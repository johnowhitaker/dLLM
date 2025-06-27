# dLLM

Messing with Diffusion LLMs

Video for context: https://www.youtube.com/watch?v=Ds_cTclxV2o

## Scripts

### Training (`train.py`)
Fine-tune ModernBERT into a diffusion-style LLM using variable masking ratios.

```bash
# Quick test run (400 steps)
python train.py \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --max-train-steps 400 \
  --log-every 100

# Full training with W&B logging and HF upload
python train.py \
  --batch-size 4 \
  --gradient-accumulation-steps 16 \
  --num-epochs 2 \
  --use-wandb \
  --wandb-project "modernbert-diffusion" \
  --upload-to-hf \
  --hf-repo-name "username/modernbert-diffusion"
```

### Sampling (`sample_model.py`)
Generate responses using greedy sampling from a trained model.

```bash
# Sample from pre-trained model
python sample_model.py \
  --model-dir "johnowhitaker/modernbert-diffusion" \
  --question "What is the meaning of life?" \
  --ans-len 32

# Sample from locally trained model
python sample_model.py \
  --model-dir "./modernbert-diffusion-finetuned" \
  --question "Tell me a fun fact about cows" \
  --ans-len 50
```
