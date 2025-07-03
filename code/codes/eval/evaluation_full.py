#!/usr/bin/env python3
import sys
import json
import argparse
from bert_score import score
import sacrebleu
from rouge_score import rouge_scorer


def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def main():
    parser = argparse.ArgumentParser(
        description="Compute BERTScore between two lists of texts (refs vs. preds)."
    )
    parser.add_argument(
        "--refs", "-r", default="/home/hoangphuc/MemoChat/data/locomo/memochat_locomo10_root.json",
        help="Path to reference file: one reference answer per line."
    )
    parser.add_argument(
        "--preds", "-p", default="/home/hoangphuc/MemoChat/data/locomo/locomo_testing/locomo_testing_t5-3b_1k_1.json",
        help="Path to prediction file: one predicted answer per line, in the same order."
    )
    parser.add_argument(
        "--model", "-m", default="microsoft/deberta-xlarge-mnli",
        help="HuggingFace model for BERTScore (default: %(default)s)."
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=4,
        help="Batch size for BERTScore computation (default: %(default)s)."
    )
    args = parser.parse_args()

    # refs  = read_lines(args.refs)
    # preds = read_lines(args.preds)
    # Đọc dữ liệu từ file JSON
    with open(args.refs, 'r', encoding='utf-8') as f:
        refs_data = json.load(f)
    with open(args.preds, 'r', encoding='utf-8') as f:
        preds_data = json.load(f)
    # Lấy các câu trả lời từ dữ liệu
    refs = []
    for item in refs_data:
        conversations = item["conversations"]
        for conversation in conversations:
            if conversation["from"] == "chatbot":
                refs.append(str(conversation["value"]))
    preds = []
    for item in preds_data:
        conversations = item["conversations"]
        for conversation in conversations:
            if conversation["from"] == "chatbot":
                preds.append(str(conversation["value"]))
    
    # kiểm tra độ dài của refs và preds
    if not refs or not preds:
        sys.exit("Error: refs or preds are empty. Please check your input files.")
    if len(refs) == 0 or len(preds) == 0:
        sys.exit("Error: refs or preds are empty. Please check your input files.")

    # refs = refs[:len(preds)]  # Giới hạn số lượng câu trả lời tham chiếu
    
    # Lưu vào 1 file json gồm các item, mỗi item có 2 trường "refs" và "preds"
    output_data = []
    for ref, pred in zip(refs, preds):
        output_data.append({
            "refs": ref,
            "preds": pred
        })
    with open('/home/hoangphuc/MemoChat/code/codes/eval/output_data.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    if len(refs) != len(preds):
        sys.exit(f"Error: refs ({len(refs)}) and preds ({len(preds)}) have different lengths.")

    # compute BERTScore
    P, R, F1 = score(
        cands=preds,
        refs=refs,
        model_type=args.model,
        batch_size=args.batch_size,
        lang="en"       # hoặc để None nếu multilingual
    )

    # --- BLEU ---
    # sacrebleu expects list of preds, list of list of refs
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score

    # --- ROUGE1, ROUGE2, ROUGEL (f1 scores) ---
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    rouge1_f, rouge2_f, rougel_f = 0.0, 0.0, 0.0
    for ref, pred in zip(refs, preds):
        scores = scorer.score(ref, pred)
        rouge1_f += scores["rouge1"].fmeasure
        rouge2_f += scores["rouge2"].fmeasure
        rougel_f += scores["rougeL"].fmeasure
    n = len(preds)
    rouge1_f = rouge1_f / n * 100
    rouge2_f = rouge2_f / n * 100
    rougel_f = rougel_f / n * 100

    # trung bình và scale sang %
    p_avg  = float(P.mean()  * 100)
    r_avg  = float(R.mean()  * 100)
    f1_avg = float(F1.mean() * 100)

    print(f"BERTScore (model={args.model}):")
    print(f"  Precision: {p_avg:.2f}%")
    print(f"  Recall:    {r_avg:.2f}%")
    print(f"  F1:        {f1_avg:.2f}%")

    # In thêm BLEU và ROUGE
    print()
    print(f"BLEU: {bleu:.2f}%")
    print(f"ROUGE-1 (F1): {rouge1_f:.2f}%")
    print(f"ROUGE-2 (F1): {rouge2_f:.2f}%")
    print(f"ROUGE-L (F1): {rougel_f:.2f}%")

if __name__ == "__main__":
    main()
