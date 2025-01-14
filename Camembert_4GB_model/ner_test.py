import torch
from torch.utils.data import DataLoader

from .camembert_for_pretraining import CamembertForPreTraining
from model.camembert_for_ner import CamembertForNER
from .ner_trainer import NERDataset, extract_entities_bio

def compute_f1_bio(pred_labels, gold_labels):
    # pareil que dans ner_trainer
    pred_ents = set(extract_entities_bio(pred_labels))
    gold_ents = set(extract_entities_bio(gold_labels))
    tp = len(pred_ents & gold_ents)
    fp = len(pred_ents - gold_ents)
    fn = len(gold_ents - pred_ents)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return tp, fp, fn, prec, rec, f1

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
    pretrained = CamembertForPreTraining.load_pretrained(base_pretrained_path, device=device)
    num_labels = len(label2id)
    model = CamembertForNER(pretrained, num_labels).to(device)

    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)

    test_data = NERDataset(test_path, tokenizer, label2id)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    prec, rec, f1 = evaluate_ner_f1(model, test_loader, id2label, device)
    print(f"[TEST - NER] F1={f1 * 100:.2f}, precision={prec * 100:.2f}, recall={rec * 100:.2f}")
    return prec, rec, f1