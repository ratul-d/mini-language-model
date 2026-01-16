def get_stats(ids):
    """ count pairs """
    counts = {}
    for pair in zip(ids,ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts

def merge(ids, pair, idx):
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

#BYTE-PAIR-ENCODING
#------------------------
def bpe(text, vocab_size):
    tokens = text.encode("utf-8")  # raw bytes
    tokens = list(map(int, tokens))  # tokens before Byte-Pair-Encoding

    print("text Length: ", len(text))
    print("tokens before BPE Length: ", len(tokens))

    #vocab_size = 276                        # the desired final vocab  size
    num_merges = vocab_size - 256           # we already have 256 tokens for the raw bytes, hence we subtract
    ids = list(tokens)

    merges = {}                             # save the pairs being merged and their index: (int,int) --> int
    for i in range(num_merges):
        stats = get_stats(ids)              # count pairs
        if not stats:
            break

        pair = max(stats, key=stats.get)    # get most occurring pair
        idx = 256 + i
        print(f"Merging {pair} into a new token: {idx}")
        ids = merge(ids, pair, idx)         # merge
        merges[pair] = idx

    print("tokens length before BPE: ", len(tokens))
    print("tokens length after BPE: ", len(ids))
    print(f"Compression Ratio: {len(tokens) / len(ids):.2f}X")

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    return merges, vocab


# ENCODE-DECODE FUNCTIONS
def encode(text, merges):
    tokens = list(text.encode("utf-8")) # tokens before Byte-Pair-Encoding
    while True:
        stats = get_stats(tokens) # count pairs
        if not stats:
            break

        pair = min(stats, key=lambda p: merges.get(p, float("inf"))) # select the pair whose merge rule was learned earliest (lowest merge id)
        if pair not in merges:
            break

        idx = merges[pair]  # get idx for pair from merges-dict
        tokens = merge(tokens, pair, idx)  # merge
    return tokens

def decode(ids, vocab):
    tokens = b"".join(vocab[idx] for idx in ids)
    text =  tokens.decode('utf-8', errors='replace')
    return text