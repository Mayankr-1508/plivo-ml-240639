import argparse
import torch
import torch.nn.functional as F
from model import GPTLanguageModel
from tokenizer import ByteTokenizer

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg = checkpoint['config']
    
    model = GPTLanguageModel(
        vocab_size=cfg['vocab_size'],
        block_size=cfg['block_size'],
        n_embd=cfg['n_embd'],
        n_head=cfg['n_head'],
        n_layer=cfg['n_layer'],
        dropout=cfg['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, cfg

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT language model.")
    parser.add_argument('--checkpoint', type=str, default='ckpt.pt', help='Path to model checkpoint')
    parser.add_argument('--text_file', type=str, required=True, help='Path to evaluation text file')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from {args.checkpoint} onto {device}...")
    
    model, cfg = load_model(args.checkpoint)
    model.to(device)
    model.eval()

    print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters()):,d} parameters.")
    
    with open(args.text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode(text)
    data = torch.tensor(tokens, dtype=torch.long, device=device)
    
    block_size = cfg['block_size']
    total_loss = 0.0
    tokens_scored = 0

    print(f"Evaluating on {args.text_file}...")
    with torch.no_grad():
        for i in range(0, len(data) - block_size, block_size):
            x = data[i:i+block_size].unsqueeze(0)
            y = data[i+1:i+1+block_size].unsqueeze(0)
            if x.size(1) < block_size:
                break
            logits, loss = model(x, y)
            total_loss += loss.item() * x.size(1)
            tokens_scored += x.size(1)

    avg_loss = total_loss / tokens_scored if tokens_scored > 0 else 0.0
    bpb = avg_loss / math.log(2) if 'math' in globals() else avg_loss / 0.693147

    print(f"\nEvaluation Results:")
    print(f"Tokens Scored: {tokens_scored}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Bits-Per-Byte (BPB): {bpb:.4f}")

if __name__ == '__main__':
    import math
    main()
