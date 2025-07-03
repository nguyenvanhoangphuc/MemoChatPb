# Tính retrieval metrics cho các câu hỏi và câu trả lời trong tập dữ liệu Locomo
# mỗi câu hỏi sẽ được model tìm các thông tin liên quan trong bộ corpus (danh sách các câu trong đoạn hội thoại), sau đó model sẽ trả lời dựa trên các thông tin này.
# Câu trả lời tương ứng với từng câu hỏi sẽ được lưu dưới dạng JSON: 
# {
#     "question": "Câu hỏi",
#     "answer": "Câu trả lời",
#     "evidence": ["Câu 2 trong đoạn hội thoại", "Câu 5 trong đoạn hội thoại"]
# }
# Tương tự trong Locomo, mỗi câu hỏi sẽ có một danh sách các câu trả lời tương ứng.
# Mỗi câu hỏi sẽ có một danh sách các câu trong đoạn hội thoại được sử dụng để trả lời câu hỏi đó.
# Các câu hỏi, câu trả lời, các câu liên quan trong đoạn hội thoại sẽ được lưu trong một file JSON với các trường "question", "answer" và "evidence".

# Tính toán các metrics như Recall@k, Precision@k, F1@k cho các câu hỏi và câu liên quan trong hội thoại trong tập dữ liệu Locomo.
import json
from collections import defaultdict

def precision_recall_f1_at_k(gt_list, pred_list, k):
    # Ensure both are sets
    gt_set = set(gt_list)
    pred_set = set(pred_list[:k])
    
    true_positive = len(gt_set & pred_set)
    
    precision = true_positive / len(pred_set) if pred_set else 0
    recall = true_positive / len(gt_set) if gt_set else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    
    return precision, recall, f1

def evaluate_metrics(gt_file, pred_file, ks=[1, 3, 5]):
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    # Match questions between GT and predicted
    pred_dict = {item['question']: item['predicted_evidence'] for item in pred_data}
    
    results = defaultdict(list)
    
    for item in gt_data:
        question = item['question']
        gt_evidence = item['evidence']
        pred_evidence = pred_dict.get(question, [])
        
        for k in ks:
            p, r, f1 = precision_recall_f1_at_k(gt_evidence, pred_evidence, k)
            results[f'Precision@{k}'].append(p)
            results[f'Recall@{k}'].append(r)
            results[f'F1@{k}'].append(f1)
    
    # Average
    for k in ks:
        print(f"\n--- Metrics @ {k} ---")
        print(f"Precision@{k}: {sum(results[f'Precision@{k}']) / len(results[f'Precision@{k}']):.4f}")
        print(f"Recall@{k}:    {sum(results[f'Recall@{k}']) / len(results[f'Recall@{k}']):.4f}")
        print(f"F1@{k}:        {sum(results[f'F1@{k}']) / len(results[f'F1@{k}']):.4f}")

# Ví dụ gọi hàm
evaluate_metrics('/home/hoangphuc/MemoChat/code/codes/eval/ground_truth.json', '/home/hoangphuc/MemoChat/code/codes/eval/model_output.json', ks=[1, 3, 5])
