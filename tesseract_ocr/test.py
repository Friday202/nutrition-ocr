import fasttext
from matplotlib import pyplot as plt

# Download once: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin


# def compute_likelihood(text, lang="sl"):
#     model = fasttext.load_model("lid.176.bin")
#     text = " ".join(text.split())
#     print(text)
#     tokens = [w.strip(".,:;!?") for w in text.split()]
#
#     for token in tokens:
#         pred = model.predict(token)
#         print(token, pred)

# def compute_likelihood(text, lang="sl"):
#     model = fasttext.load_model("lid.176.bin")
#     text = " ".join(text.split())
#     tokens = [w.strip(".,:;!?") for w in text.split()]
#     for token in tokens:
#         labels, probs = model.predict(token, k=50)
#         print(labels, probs)

def compute_likelihood(text, lang="sl"):
    model = fasttext.load_model("lid.176.bin")
    text = " ".join(text.split())
    tokens = [w.strip(".,:;!?") for w in text.split()]

    probs_sl = []
    for token in tokens:
        labels, probs = model.predict(token, k=1)

        # check if __label__sl is in labels
        if f"__label__{lang}" in labels:
            idx = labels.index(f"__label__{lang}")
            probs_sl.append(probs[idx])
        else:
            probs_sl.append(0.0)

    # plot distribution
    plt.figure(figsize=(12, 6))
    plt.bar(tokens, probs_sl, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(f"Probability of __label__{lang}")
    plt.title(f"Language likelihood per token for {lang.upper()}")
    plt.show()
