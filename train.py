#!/usr/bin/env python3
"""
ModernBERT Diffusion LLM Training Script

Fine-tunes ModernBERT into a diffusion-style LLM using variable masking ratios
on instruction data, similar to the LLADA approach.
"""

import os
import random
import itertools
import math
import torch
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW
from datasets import load_dataset
from tqdm.auto import tqdm
from itertools import islice

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for logging.")


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    # Model configuration
    model_id: str = "answerdotai/ModernBERT-large"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    
    # Dataset configuration
    dataset_name: str = "allenai/tulu-3-sft-mixture-0225"
    dataset_split: str = "train"
    dataset_cache_dir: str = "./data"
    max_length: int = 512
    mask_ratio_min: float = 0.15
    mask_ratio_max: float = 0.99
    train_split_ratio: float = 0.95
    num_proc: int = 32
    
    # Training configuration
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 1
    max_train_steps: Optional[int] = None  # Override num_epochs if set
    warmup_ratio: float = 0.06
    
    # Evaluation and logging
    log_every: int = 200
    eval_batches: int = 8
    save_dir: str = "modernbert-diffusion-finetuned"
    
    # W&B configuration
    use_wandb: bool = False
    wandb_project: str = "modernbert-diffusion"
    wandb_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    
    # HuggingFace upload
    upload_to_hf: bool = False
    hf_repo_name: Optional[str] = None
    hf_token: Optional[str] = None
    hf_private: bool = True


def parse_args() -> TrainingConfig:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train ModernBERT Diffusion LLM")
    
    # Model configuration
    parser.add_argument("--model-id", type=str, default="answerdotai/ModernBERT-large",
                        help="HuggingFace model ID to use as base model")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", 
                        choices=["float32", "float16", "bfloat16"],
                        help="Torch dtype for model weights")
    parser.add_argument("--device-map", type=str, default="auto",
                        help="Device mapping strategy")
    
    # Dataset configuration
    parser.add_argument("--dataset-name", type=str, default="allenai/tulu-3-sft-mixture-0225",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset-split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--dataset-cache-dir", type=str, default="./data",
                        help="Directory to cache dataset")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--mask-ratio-min", type=float, default=0.15,
                        help="Minimum masking ratio")
    parser.add_argument("--mask-ratio-max", type=float, default=0.99,
                        help="Maximum masking ratio")
    parser.add_argument("--train-split-ratio", type=float, default=0.95,
                        help="Ratio of data to use for training (rest for validation)")
    parser.add_argument("--num-proc", type=int, default=32,
                        help="Number of processes for dataset preprocessing")
    
    # Training configuration
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max-train-steps", type=int, default=None,
                        help="Maximum number of training steps (overrides num_epochs if set)")
    parser.add_argument("--warmup-ratio", type=float, default=0.06,
                        help="Warmup ratio of total steps")
    
    # Evaluation and logging
    parser.add_argument("--log-every", type=int, default=200,
                        help="Log metrics every N steps")
    parser.add_argument("--eval-batches", type=int, default=8,
                        help="Number of batches to use for validation")
    parser.add_argument("--save-dir", type=str, default="modernbert-diffusion-finetuned",
                        help="Directory to save the trained model")
    
    # W&B configuration
    parser.add_argument("--use-wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="modernbert-diffusion",
                        help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity name")
    
    # HuggingFace upload
    parser.add_argument("--upload-to-hf", action="store_true",
                        help="Upload trained model to HuggingFace Hub")
    parser.add_argument("--hf-repo-name", type=str, default=None,
                        help="HuggingFace repository name (required if --upload-to-hf)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token (if not set, will try to use cached token)")
    parser.add_argument("--hf-private", action="store_true",
                        help="Make HuggingFace repository private")
    
    args = parser.parse_args()
    
    # Convert to TrainingConfig
    config = TrainingConfig(
        model_id=args.model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dataset_cache_dir=args.dataset_cache_dir,
        max_length=args.max_length,
        mask_ratio_min=args.mask_ratio_min,
        mask_ratio_max=args.mask_ratio_max,
        train_split_ratio=args.train_split_ratio,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        max_train_steps=args.max_train_steps,
        warmup_ratio=args.warmup_ratio,
        log_every=args.log_every,
        eval_batches=args.eval_batches,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
        upload_to_hf=args.upload_to_hf,
        hf_repo_name=args.hf_repo_name,
        hf_token=args.hf_token,
        hf_private=args.hf_private,
    )
    
    # Validation
    if config.upload_to_hf and not config.hf_repo_name:
        raise ValueError("--hf-repo-name is required when --upload-to-hf is set")
    
    if config.use_wandb and not WANDB_AVAILABLE:
        raise ValueError("wandb is not installed. Install with 'pip install wandb'")
    
    return config


def setup_model_and_tokenizer(config: TrainingConfig) -> Tuple[Any, Any, Dict[str, int]]:
    """Load model and tokenizer"""
    print(f"Loading model and tokenizer: {config.model_id}")
    
    # Setup dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[config.torch_dtype]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    
    # Load model
    model = AutoModelForMaskedLM.from_pretrained(
        config.model_id,
        torch_dtype=torch_dtype,
        device_map=config.device_map,
        low_cpu_mem_usage=True,
    )
    
    # Get special token IDs
    special_tokens = {
        "mask_id": tokenizer.mask_token_id,
        "cls_id": tokenizer.cls_token_id,
        "sep_id": tokenizer.sep_token_id,
        "pad_id": tokenizer.pad_token_id,
    }
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Special tokens: {special_tokens}")
    
    return model, tokenizer, special_tokens


def create_dataset(config: TrainingConfig, tokenizer: Any, special_tokens: Dict[str, int]):
    """Load and preprocess dataset"""
    print(f"Loading dataset: {config.dataset_name}")
    
    # Load raw dataset
    raw_ds = load_dataset(
        config.dataset_name,
        split=config.dataset_split,
        cache_dir=config.dataset_cache_dir
    )
    print(f"Raw dataset length: {len(raw_ds):,}")
    
    def join_dialogue(msgs):
        """Convert messages to flat string with SEP boundaries"""
        if len(msgs) < 2:
            return None
        u = msgs[0]["content"].strip()
        a = msgs[1]["content"].strip()
        return f"User: {u} {tokenizer.sep_token} Assistant: {a}"
    
    def apply_random_mask(example):
        """Apply random masking to assistant responses"""
        text = join_dialogue(example["messages"])
        if text is None:
            # Return dummy data for invalid examples
            dummy_ids = [special_tokens["cls_id"]] + [special_tokens["pad_id"]] * (config.max_length - 1)
            return {
                "input_ids": dummy_ids,
                "attention_mask": [1] + [0] * (config.max_length - 1),
                "labels": [-100] * config.max_length
            }
        
        enc = tokenizer(
            text,
            truncation=True,
            max_length=config.max_length,
            padding="max_length"
        )
        ids = enc["input_ids"]
        labels = [-100] * len(ids)  # -100 -> ignored by CE loss
        
        # Find assistant region (everything after first [SEP])
        if special_tokens["sep_id"] not in ids:
            return {**enc, "labels": labels}
        
        sep_pos = ids.index(special_tokens["sep_id"])  # first [SEP]
        cand = [i for i in range(sep_pos + 1, len(ids))
                if ids[i] not in (special_tokens["pad_id"], 
                                 special_tokens["cls_id"], 
                                 special_tokens["sep_id"])]
        
        if not cand:
            return {**enc, "labels": labels}
        
        # Variable mask ratio
        m_ratio = random.uniform(config.mask_ratio_min, config.mask_ratio_max)
        n_mask = max(1, int(len(cand) * m_ratio))
        chosen = random.sample(cand, n_mask)
        
        for idx in chosen:
            labels[idx] = ids[idx]  # remember ground truth
            dice = random.random()
            if dice < 0.8:  # 80%
                ids[idx] = special_tokens["mask_id"]
            elif dice < 0.9:  # 10%
                ids[idx] = random.randint(0, tokenizer.vocab_size - 1)
            # else leave token unchanged (10%)
        
        enc["input_ids"] = ids
        enc["labels"] = labels
        return enc
    
    # Process dataset
    print("Processing dataset with masking...")
    proc_ds = raw_ds.map(
        apply_random_mask, 
        remove_columns=raw_ds.column_names, 
        num_proc=config.num_proc
    )
    proc_ds.set_format(type="torch")
    
    # Split train/val
    shuffled_ds = proc_ds.shuffle(seed=42)
    train_size = int(config.train_split_ratio * len(shuffled_ds))
    
    train_ds = shuffled_ds.select(range(train_size))
    val_ds = shuffled_ds.select(range(train_size, len(shuffled_ds)))
    
    print(f"Train dataset: {len(train_ds):,} examples")
    print(f"Validation dataset: {len(val_ds):,} examples")
    
    return train_ds, val_ds


def accuracy_buckets(logits: torch.Tensor, labels: torch.Tensor, attn: torch.Tensor) -> Tuple[float, List[float]]:
    """
    Calculate accuracy in buckets by masking ratio
    Returns:
        global_acc (float): Overall accuracy
        bucket_acc (List[float]): Accuracy for ≤.25, .25-.5, .5-.75, >.75 mask ratios
    """
    with torch.no_grad():
        pred = logits.argmax(-1)
        mask = labels != -100  # only masked positions count
        correct = (pred == labels) & mask
        
        # Global accuracy
        tot_masked = mask.sum().item()
        tot_corr = correct.sum().item()
        global_acc = tot_corr / tot_masked if tot_masked else 0.0
        
        # Buckets by sample-level mask ratio
        bucket_corr = [0, 0, 0, 0]
        bucket_total = [0, 0, 0, 0]
        edges = (0.25, 0.50, 0.75, 1.01)  # last edge slightly >1
        
        for b in range(labels.size(0)):
            n_mask = mask[b].sum().item()
            if n_mask == 0:  # should be rare
                continue
            # denominator = real tokens (ignore pads)
            seq_len = attn[b].sum().item()
            ratio = n_mask / seq_len
            # bucket index
            for i, edge in enumerate(edges):
                if ratio <= edge:
                    bucket_total[i] += n_mask
                    bucket_corr[i] += correct[b].sum().item()
                    break
        
        bucket_acc = [c / t if t else 0.0 for c, t in zip(bucket_corr, bucket_total)]
    
    return global_acc, bucket_acc


@torch.no_grad()
def evaluate_model(model: Any, val_loader: DataLoader, config: TrainingConfig, device: str) -> Tuple[float, float, List[float]]:
    """Evaluate model on validation set"""
    model.eval()
    tot_loss, tot_acc, bucket_hits = 0., 0., [0, 0, 0, 0]
    
    for batch in islice(val_loader, config.eval_batches):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss.item()
        tot_loss += loss
        acc, bucket_acc = accuracy_buckets(out.logits, batch["labels"], batch["attention_mask"])
        tot_acc += acc
        bucket_hits = [h + a for h, a in zip(bucket_hits, bucket_acc)]
    
    n = config.eval_batches
    val_loss = tot_loss / n
    val_acc = tot_acc / n
    bucket_acc = [b / n for b in bucket_hits]
    
    return val_loss, val_acc, bucket_acc


def setup_training(model: Any, config: TrainingConfig, train_ds_len: int) -> Tuple[Any, Any, int]:
    """Setup optimizer and scheduler"""
    optimizer = AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    steps_per_epoch = len(range(0, train_ds_len, config.batch_size))
    
    # Use max_train_steps if provided, otherwise calculate from epochs
    if config.max_train_steps is not None:
        total_steps = config.max_train_steps
        max_epochs = max(1, (total_steps * config.gradient_accumulation_steps) // steps_per_epoch + 1)
    else:
        total_steps = config.num_epochs * steps_per_epoch // config.gradient_accumulation_steps
        max_epochs = config.num_epochs
    
    warmup_steps = int(config.warmup_ratio * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        warmup_steps, 
        total_steps
    )
    
    print(f"Training setup:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    if config.max_train_steps is not None:
        print(f"  Using max_train_steps override: {config.max_train_steps}")
    
    return optimizer, scheduler, max_epochs


def train_model(model: Any, tokenizer: Any, train_ds, val_ds, config: TrainingConfig):
    """Main training loop"""
    device = next(model.parameters()).device
    
    # Setup data loaders
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    
    # Setup training
    optimizer, scheduler, max_epochs = setup_training(model, config, len(train_ds))
    
    # Initialize W&B if requested
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            entity=config.wandb_entity,
            config=config.__dict__
        )
    
    # Training state
    global_step = 0
    best_val_loss = float('inf')
    
    # Training loop
    model.train()
    for epoch in range(max_epochs):
        print(f"\n=== Epoch {epoch + 1}/{max_epochs} ===")
        
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch + 1}", 
            leave=False, 
            dynamic_ncols=True
        )
        
        running_loss = 0.0
        running_acc = 0.0
        optimizer.zero_grad()
        
        for step, batch in enumerate(pbar, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            out = model(**batch)
            loss = out.loss / config.gradient_accumulation_steps
            loss.backward()
            
            # Metrics
            acc, _ = accuracy_buckets(out.logits.detach(), batch["labels"], batch["attention_mask"])
            running_loss += out.loss.item()
            running_acc += acc
            
            # Gradient accumulation
            if step % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log to W&B
                if config.use_wandb:
                    wandb.log({
                        "train/loss": out.loss.item(),
                        "train/accuracy": acc,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/step": global_step,
                    })
            
            # Update progress bar
            if step % 20 == 0 or step == 1:
                pbar.set_postfix(
                    loss=running_loss / step,
                    acc=running_acc / step,
                    lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )
            
            # Check if we've reached max steps
            if config.max_train_steps is not None and global_step >= config.max_train_steps:
                print(f"\n🏁 Reached max_train_steps ({config.max_train_steps}), stopping training")
                break
            
            # Validation
            if global_step % config.log_every == 0:
                val_loss, val_acc, val_buckets = evaluate_model(model, val_loader, config, device)
                
                print(f"\n🧮 Step {global_step:6d} | "
                      f"train_loss {running_loss/step:.4f}  "
                      f"train_acc {running_acc/step:.3f} | "
                      f"val_loss {val_loss:.4f}  val_acc {val_acc:.3f} | "
                      f"bucket_acc {['{:.3f}'.format(x) for x in val_buckets]}")
                
                # Log to W&B
                if config.use_wandb:
                    wandb.log({
                        "val/loss": val_loss,
                        "val/accuracy": val_acc,
                        "val/bucket_acc_0-25": val_buckets[0],
                        "val/bucket_acc_25-50": val_buckets[1],
                        "val/bucket_acc_50-75": val_buckets[2],
                        "val/bucket_acc_75+": val_buckets[3],
                        "train/step": global_step,
                    })
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"💾 New best validation loss: {val_loss:.4f}")
                    save_model(model, tokenizer, config, suffix="_best")
                
                model.train()  # Back to train mode
        
        # Break out of epoch loop if max steps reached
        if config.max_train_steps is not None and global_step >= config.max_train_steps:
            break
    
    # Final save
    save_model(model, tokenizer, config)
    
    # Upload to HuggingFace if requested
    if config.upload_to_hf:
        upload_to_huggingface(model, tokenizer, config)
    
    # Finish W&B
    if config.use_wandb:
        wandb.finish()


def save_model(model: Any, tokenizer: Any, config: TrainingConfig, suffix: str = ""):
    """Save model and tokenizer"""
    save_path = Path(config.save_dir + suffix)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving model to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def upload_to_huggingface(model: Any, tokenizer: Any, config: TrainingConfig):
    """Upload model to HuggingFace Hub"""
    print(f"🚀 Uploading to HuggingFace: {config.hf_repo_name}")
    
    try:
        model.push_to_hub(
            config.hf_repo_name,
            token=config.hf_token,
            private=config.hf_private,
            commit_message="Add diffusion-style fine-tuned ModernBERT weights",
        )
        tokenizer.push_to_hub(
            config.hf_repo_name,
            token=config.hf_token,
        )
        print(f"✅ Successfully uploaded to https://huggingface.co/{config.hf_repo_name}")
    except Exception as e:
        print(f"❌ Failed to upload to HuggingFace: {e}")


def main():
    """Main training function"""
    # Parse arguments
    config = parse_args()
    
    print("🚀 Starting ModernBERT Diffusion LLM Training")
    print(f"Configuration: {config}")
    
    # Setup model and tokenizer
    model, tokenizer, special_tokens = setup_model_and_tokenizer(config)
    
    # Create dataset
    train_ds, val_ds = create_dataset(config, tokenizer, special_tokens)
    
    # Train model
    train_model(model, tokenizer, train_ds, val_ds, config)
    
    print("🎉 Training completed!")


if __name__ == "__main__":
    main()
