from tokenizer.bpe_tokenizer import BPETokenizer

# LOAD DATASET
with open("Dataset/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

vocab_size=300
tok = BPETokenizer(vocab_size)
tok.train(text,verbose=True)

tok.save("models/BPE-Transformer/shakespeare_tokenizer.pkl")
