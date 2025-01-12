from fine_tuning.parsing_trainer import train_parsing
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer

def grid_search_parsing(
    pretrained_path,
    train_path,
    dev_path,
    tokenizer,
    rel2id,
    n_rels,
    device="cuda"
):
    # On fixe un ensemble de lrs, batch_sizes
    lrs=[1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4, 3e-4, 5e-4]
    batch_sizes=[16,32]

    best_las=-1.0
    best_config=None
    best_ckpt=None

    for lr in lrs:
        for bs in batch_sizes:
            ckpt_name=f"parsing_lr{lr}_bs{bs}"
            print(f"\n=== GRID Parsing: lr={lr}, bs={bs}, epochs=30 ===")
            las= train_parsing(
                pretrained_path=pretrained_path,
                train_path=train_path,
                dev_path=dev_path,
                tokenizer=tokenizer,
                rel2id=rel2id,
                n_rels=n_rels,
                lr=lr,
                epochs=30,
                batch_size=bs,
                device=device,
                out_model_path=ckpt_name
            )
            if las>best_las:
                best_las=las
                best_config=(lr, bs)
                best_ckpt = ckpt_name

    # On loggue comme TensorBoard ne marche pas
    print(f"BEST LAS on dev= {best_las*100:.2f}% with (lr={best_config[0]}, bs={best_config[1]}) => {best_ckpt}")

if __name__=="__main__":
    tokenizer=SentencePieceTokenizer("data/processed/spm.model")
    rel2id = {
        '_': 0, 'acl': 1, 'acl:relcl': 2, 'advcl': 3, 'advcl:cleft': 4, 'advmod': 5, 'amod': 6,
        'appos': 7, 'aux': 8, 'aux:caus': 9, 'aux:pass': 10, 'aux:tense': 11, 'case': 12,
        'cc': 13, 'ccomp': 14, 'compound': 15, 'conj': 16, 'cop': 17, 'csubj': 18,
        'csubj:pass': 19, 'dep': 20, 'dep:comp': 21, 'det': 22, 'discourse': 23,
        'dislocated': 24, 'expl:comp': 25, 'expl:pass': 26, 'expl:pv': 27, 'expl:subj': 28,
        'fixed': 29, 'flat': 30, 'flat:foreign': 31, 'flat:name': 32, 'goeswith': 33,
        'iobj': 34, 'iobj:agent': 35, 'mark': 36, 'nmod': 37, 'nsubj': 38, 'nsubj:caus': 39,
        'nsubj:outer': 40, 'nsubj:pass': 41, 'nummod': 42, 'obj': 43, 'obj:agent': 44,
        'obj:lvc': 45, 'obl': 46, 'obl:agent': 47, 'obl:arg': 48, 'obl:mod': 49,
        'orphan': 50, 'parataxis': 51, 'parataxis:insert': 52, 'punct': 53, 'root': 54,
        'vocative': 55, 'xcomp': 56
    }
    n_rels=len(rel2id)

    grid_search_parsing(
        pretrained_path="checkpoints/camembert_pretrained.pt",
        train_path="data/tasks/parsing/train.conllu",
        dev_path="data/tasks/parsing/dev.conllu",
        tokenizer=tokenizer,
        rel2id=rel2id,
        n_rels=n_rels,
        device="cuda"
    )