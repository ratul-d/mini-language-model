import pickle
import os

class BPETokenizer:
    def __init__(self, vocab_size=256):
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")
        self.target_vocab_size = vocab_size
        self.merges={}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def _get_stats(self, ids):
        """ count pairs """
        counts = {}
        for pair in zip(ids,ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1

        return counts

    def _merge(self,ids, pair, idx):
        """ function to replace pairs with new idx """
        newids=[]
        i=0

        while i < len(ids):
            # if we are not at very last position AND the pair matches, replace it
            if i < len(ids) - 1 and ids[i]==pair[0] and ids[i+1]==pair[1]:
                newids.append(idx)
                i+=2
            else:
                newids.append(ids[i])
                i+=1

        return newids

    # BYTE-PAIR-ENCODING
    def train(self, text, verbose=False):
        tokens = text.encode("utf-8")  # raw bytes
        tokens = list(map(int, tokens))  # tokens before Byte-Pair-Encoding

        print("text Length: ", len(text))
        print("tokens length before BPE: ", len(tokens))

        # vocab_size = 276                        # the desired final vocab  size
        num_merges = self.target_vocab_size - 256  # we already have 256 tokens for the raw bytes, hence we subtract
        ids = list(tokens)

        for i in range(num_merges):
            stats = self._get_stats(ids)  # count pairs
            if not stats:
                break

            pair = max(stats, key=stats.get)  # get most occurring pair
            idx = 256 + i
            if verbose:
                print(f"Merging {pair} into a new token: {idx}")

            ids = self._merge(ids, pair, idx)  # merge
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        print("tokens length before BPE: ", len(tokens))
        print("tokens length after BPE: ", len(ids))
        print(f"Compression Ratio: {len(tokens) / len(ids):.2f}X")

        return self

    # ENCODE-DECODE FUNCTIONS
    def encode(self, text):
        tokens = list(text.encode("utf-8"))  # tokens before Byte-Pair-Encoding

        # apply merges in training order
        for pair, idx in self.merges.items():
            tokens = self._merge(tokens, pair, idx)

        return tokens

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode('utf-8', errors='replace')
        return text



    def save(self, path):
        """Save BPE merges and vocab to disk."""
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "merges": self.merges,
                "vocab": self.vocab,
                "target_vocab_size": self.target_vocab_size
            }, f, )

        print("Saved")

    @classmethod
    def load(cls, path):
        """Load BPE merges and vocab from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        tok = cls(vocab_size=data["target_vocab_size"])
        tok.merges  = data["merges"]
        tok.vocab  = data["vocab"]

        return tok

    @property
    def vocab_size(self):
        return len(self.vocab)