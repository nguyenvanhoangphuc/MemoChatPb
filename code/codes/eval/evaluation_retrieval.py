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
import sys

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

    conversations_list = [item['conversations'] for item in pred_data]
    qa_list = []
    for conversations in conversations_list:
        qa_list.extend(conversations)
    pred_dict = {}
    question = ""
    related_evidence = []
    dem = 0
    for qa in qa_list:
        # print("===="*20)
        # print("qa:", qa)
        # print("===="*20)
        if qa['from'] == 'human':
            qa['value'] = qa['value'].strip()
            question = qa['value']
        elif qa['from'] == 'chatbot':
            thinking = qa.get('thinking', '')
            thinking = thinking.strip()
            if not thinking:
                print(f"Empty thinking for question: {question}")
                continue
            else: 
                dem += 1
                print(dem)
            # chuyển thành json
            try:
                thinking = json.loads(thinking)
            except json.JSONDecodeError:
                print(f"Error decoding JSON for thinking: {thinking}")
                sys.exit(1)
            answer = thinking.get('answer', '')
            related_dialogs = []
            for dialogs in answer.get('related_dialogs', ""): 
                related_dialogs.extend(dialogs.split(" ### "))
            related_evidence = related_dialogs + answer.get('recent_dialogs', [])
            related_evidence = [e.strip() for e in related_evidence if e.strip()]
            pred_dict[question] = related_evidence

    # print một mẫu của pred_dict
    print("Sample of pred_dict:", list(pred_dict.items())[:1])
    print("Number of questions in pred_dict:", len(pred_dict))
        
    # Match questions between GT and predicted
    # pred_dict = {item['question']: item['predicted_evidence'] for item in pred_data}

    # xử lý gt_data
    gt_conversations_list = [item['conversations'] for item in gt_data]
    gt_qa_list = []
    for conversations in gt_conversations_list:
        gt_qa_list.extend(conversations)
        
    gt_dict = {}
    for qa in gt_qa_list:
        if qa['from'] == 'human':
            qa['value'] = qa['value'].strip()
            question = qa['value']
        elif qa['from'] == 'chatbot':
            related_evidence = qa['evidence_texts']
            gt_dict[question] = related_evidence

    results = defaultdict(list)

    # lưu gt_dict và pred_dict vào file
    with open('gt_dict.json', 'w', encoding='utf-8') as f:
        json.dump(gt_dict, f, ensure_ascii=False, indent=4)
    with open('pred_dict.json', 'w', encoding='utf-8') as f:
        json.dump(pred_dict, f, ensure_ascii=False, indent=4)
    
    for question, gt_evidence in gt_dict.items():
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
# evaluate_metrics('/home/hoangphuc/MemoChat/code/codes/eval/ground_truth.json', '/home/hoangphuc/MemoChat/code/codes/eval/model_output.json', ks=[1, 3, 5])
evaluate_metrics('/home/hoangphuc/MemoChat/data/locomo/memochat_locomo10.json', '/home/hoangphuc/MemoChat/data/locomo/locomo_testing/locomo_temp_qwen-memochat.json', ks=[100])
