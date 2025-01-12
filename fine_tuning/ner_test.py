import torch
from torch.utils.data import DataLoader

from model.camembert_for_pretraining import CamembertForPreTraining
from model.camembert_for_ner import CamembertForNER
from fine_tuning.ner_trainer import NERDataset, compute_f1_bio

def evaluate_ner_f1(model, test_loader, id2label, device='cuda'):
    model.eval()
    tp_all, fp_all, fn_all = 0,0,0

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits, _ = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1)

            bsz = input_ids.size(0)
            for b in range(bsz):
                length = attention_mask[b].sum().item()
                gold_seq = labels[b, :length].cpu().tolist()
                pred_seq = preds[b, :length].cpu().tolist()

                # on filtre -100
                filtered_gold = []
                filtered_pred = []
                for g, p in zip(gold_seq, pred_seq):
                    if g != -100:
                        filtered_gold.append(g)
                        filtered_pred.append(p)
                gold_labels = [id2label[gg] for gg in filtered_gold]
                pred_labels = [id2label[pp] for pp in filtered_pred]

                tp, fp, fn, _, _, _ = compute_f1_bio(pred_labels, gold_labels)
                tp_all += tp
                fp_all += fp
                fn_all += fn

    prec = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    rec = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    model.train()
    return prec, rec, f1

def test_ner(
    best_model_path,
    base_pretrained_path,
    test_path,
    tokenizer,
    label2id,
    id2label,
    batch_size=16,
    device='cuda'
):
    """
    On charge le modèle fine-tuné et on l'évalue sur la base de test
    """
    pretrained = CamembertForPreTraining.load_pretrained(base_pretrained_path, device=device)
    num_labels = len(label2id)
    model = CamembertForNER(pretrained, num_labels).to(device)

    # On charge les poids du modèle fine-tuné
    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)

    # On charge le dataset de test
    test_data = NERDataset(test_path, tokenizer, label2id, max_len=512)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    prec, rec, f1 = evaluate_ner_f1(model, test_loader, id2label, device)
    print(f"[TEST - NER] F1={f1 * 100:.2f}, precision={prec * 100:.2f}, recall={rec * 100:.2f}")
    return prec, rec, f1