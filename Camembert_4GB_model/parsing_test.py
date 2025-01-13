import torch
from torch.utils.data import DataLoader

from .camembert_for_pretraining import CamembertForPreTraining
from model.camembert_for_parsing import CamembertForParsing
from .parsing_trainer import ParsingDataset, evaluate_parsing

def test_parsing(
    best_mode_path,
    base_pretrained_path,
    test_path,
    tokenizer,
    rel2id,
    arc_dim=512,
    rel_dim=512,
    n_rels=30,
    batch_size=16,
    device='cuda'
):
    # On charge le modèle pré-entraîné
    pretrained = CamembertForPreTraining.load_pretrained(base_pretrained_path, device=device)
    # On crée le modèle complet
    model = CamembertForParsing(pretrained, arc_dim=arc_dim, rel_dim=rel_dim, n_rels=n_rels, dropout=0.2).to(device)

    state = torch.load(best_mode_path, map_location=device)
    model.load_state_dict(state)

    test_data = ParsingDataset(test_path, tokenizer, rel2id)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    uas, las = evaluate_parsing(model, test_loader, device)
    print(f"[TEST - Parsing] UAS={uas*100:.2f}, LAS={las*100:.2f}")
    return uas, las