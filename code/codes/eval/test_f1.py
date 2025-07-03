import json
from collections import defaultdict

def precision_recall_f1_at_k(gt_list, pred_list, k):
    gt_set = set(gt_list)
    pred_set = set(pred_list[:k])
    
    tp = len(gt_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(gt_set) if gt_set else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def evaluate_retrieval_metrics(gt_file, pred_file, ks=[1, 3, 5]):
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    results = defaultdict(list)
    
    for question, gt_evidences in gt_data.items():
        pred_evidences = pred_data.get(question, [])
        
        for k in ks:
            p, r, f1 = precision_recall_f1_at_k(gt_evidences, pred_evidences, k)
            results[f'Precision@{k}'].append(p)
            results[f'Recall@{k}'].append(r)
            results[f'F1@{k}'].append(f1)

    print("=== Evaluation Results ===")
    for k in ks:
        precision_avg = sum(results[f'Precision@{k}']) / len(results[f'Precision@{k}']) if results[f'Precision@{k}'] else 0
        recall_avg = sum(results[f'Recall@{k}']) / len(results[f'Recall@{k}']) if results[f'Recall@{k}'] else 0
        f1_avg = sum(results[f'F1@{k}']) / len(results[f'F1@{k}']) if results[f'F1@{k}'] else 0
        print(f"\n@{k}")
        print(f"Precision: {precision_avg:.4f}")
        print(f"Recall:    {recall_avg:.4f}")
        print(f"F1:        {f1_avg:.4f}")

# Cách sử dụng:
evaluate_retrieval_metrics("/home/hoangphuc/MemoChat/gt_dict.json", "/home/hoangphuc/MemoChat/pred_dict.json", ks=[1, 3, 5])
