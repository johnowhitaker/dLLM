#!/usr/bin/env python3
"""
Sample model using greedy approach with specified input prompt
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_model_and_tokenizer(model_dir: str):
    """Load model and tokenizer from specified directory"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_dir)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer

def greedy_sample(model, tokenizer, question: str, ans_len: int):
    """Generate response using greedy sampling approach"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    SEP_ID, CLS_ID, MASK_ID = tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.mask_token_id

    # Construct the prompt in the proper format
    prompt = f"User: {question} {tokenizer.sep_token} Assistant:"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = [CLS_ID] + prompt_ids + [SEP_ID] + [MASK_ID] * ans_len + [SEP_ID]

    # Run the model
    with torch.no_grad():
        for i in range(ans_len):
            outs = model(input_ids=torch.tensor([ids]).to(device)).logits
            out_probs = torch.softmax(outs[0], dim=-1)
            mask_locs = (torch.tensor(ids) == MASK_ID).nonzero(as_tuple=True)[0]
            new_probs = torch.zeros_like(out_probs)
            new_probs[mask_locs] = out_probs[mask_locs]
            max_probs, max_locs = new_probs.max(dim=-1)
            max_loc = max_probs.argmax(dim=-1)
            ids[max_loc] = new_probs[max_loc].argmax().item()

    return tokenizer.decode(ids, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Sample responses from a trained model")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory of the trained model")
    parser.add_argument("--question", type=str, required=True, help="Question to ask the model")
    parser.add_argument("--ans-len", type=int, default=32, help="Length of the answer to generate")
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_dir}")
    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    
    print(f"Question: {args.question}")
    print("Generating response...")
    response = greedy_sample(model, tokenizer, args.question, args.ans_len)
    print(f"Full response: {response}")
    
    # Extract just the assistant's response part
    if "Assistant:" in response:
        assistant_response = response.split("Assistant:")[1].strip()
        print(f"Assistant response: {assistant_response}")

if __name__ == "__main__":
    main()
