import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from fine_tuning.nli_trainer import train_nli
from tokenization.sentencepiece_tokenizer import SentencePieceTokenizer

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def grid_search_nli(
    rank,
    world_size,
    model_path,
    train_path,
    dev_path,
    tokenizer,
    label2id,
    num_labels,
):
    setup(rank, world_size)
    device=f'cuda:{rank}'
    torch.cuda.set_device(device)
    # hyperparamètres
    lrs=[3e-5]
    batch_sizes=[16]

    best_acc=-1.0
    best_config=None
    best_ckpt=None

    for lr in lrs:
        for bs in batch_sizes:
            ckpt_name=f"nli_lr{lr}_bs{bs}.pt"
            if rank == 0:
                print(f"\n=== GRID NLI: lr={lr}, bs={bs}, epochs=3 ===")

            dev_acc= train_nli(
                model_path=model_path,
                train_path=train_path,
                dev_path=dev_path,
                tokenizer=tokenizer,
                label2id=label2id,
                num_labels=num_labels,
                epochs=3,
                lr=lr,
                batch_size=bs,
                device=device,
                rank=rank,
                world_size=world_size,
                out_model_path=ckpt_name if rank == 0 else None,
            )
            if rank == 0:
                print(f"[GRID] dev_acc={dev_acc*100:.2f}% => (lr={lr}, bs={bs})")

            if dev_acc>best_acc:
                best_acc=dev_acc
                best_config=(lr, bs)
                best_ckpt = ckpt_name

    if rank == 0:
        print("\n=====================")
        print(f"BEST ACC on dev= {best_acc*100:.2f}% with config (lr={best_config[0]}, bs={best_config[1]})")
        print(f"Checkpoint: {best_ckpt}")

    cleanup()

if __name__=="__main__":
    world_size = 4
    tokenizer=SentencePieceTokenizer("data/processed/spm.model")
    label2id={"entailment":0,"neutral":1,"contradiction":2}
    num_labels= len(label2id)

    mp.spawn(
        grid_search_nli,
        args=(
            world_size,
            "camembert_pretrained_4gb (1).pt",
            "data/tasks/XNLI-1.0/multinli.train.fr.tsv",
            "data/tasks/XNLI-1.0/xnli.dev.fr.tsv",
            tokenizer,
            label2id,
            num_labels,
    ),
    nprocs=world_size
)