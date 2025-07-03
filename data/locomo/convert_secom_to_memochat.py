import json
import sys

# Input/output paths
input_path = "/home/hoangphuc/MemoChat/data/locomo/secom_locomo10_new.json"     # File gốc với các trường: history, qa_with_evidence, speakers
output_path = "/home/hoangphuc/MemoChat/data/locomo/memochat_locomo10_new.json"  # File xuất ra

# Load data
with open(input_path, "r") as f:
    data = json.load(f)

# Tạo output list
output = []

for idx, entry in enumerate(data):
    new_item = {
        "id": f"mt-bench-plus-{idx}",
        "conversations": [],
        "type": "continuation"
    }

    # Lấy lịch sử hội thoại cuối cùng trong list "history"
    # Đây là dạng hội thoại có cập nhật cả hai lượt nói mỗi dòng
    # Mỗi dòng: "<Turn n>: [A]: ... \n [B]: ..."
    if not entry["history"]:
        continue

    # latest_dialogue = entry["history"][-1]  # lấy đoạn hội thoại mới nhất
    latest_dialogue = entry["history"]  # lấy đoạn hội thoại
    # latest_dialogue đang là mảng 2 chiều nối lại thành mảng 1 chiều
    latest_dialogue = [turn for sublist in latest_dialogue for turn in sublist]
    for turn_idx, turn in enumerate(latest_dialogue):
        turn = turn.replace("\n\n", "")  # Loại bỏ các dòng trống
        turn_lines = turn.split("\n")

        for line in turn_lines:
            if not line.strip():
                continue
            # Phân tích người nói
            if line.startswith("["):
                speaker = line.split("]:")[0][1:]
                try:
                    message = (line.split("]:", 1)[1]).strip()
                except Exception as e:
                    print("line:", line)
                # role = "human" if speaker == entry["speakers"][0] else "chatbot"
                new_item["conversations"].append({
                    "from": speaker,
                    "value": message,
                    "turn-info": f"{speaker}-turn-{turn_idx}"
                })
    turn_idx += 1
    # print('new_item["conversations"]:', len(new_item["conversations"]))
    # Nếu số lượt nói ở trên là lẻ thì thêm một lượt nói cuối cùng từ người nói khác
    if len(latest_dialogue) % 2 == 1:
        # print("Warning: Odd number of turns in the latest dialogue, adding a final turn.")
        # print("Latest dialogue:", latest_dialogue[-1])
        # Latest dialogue: <Turn 15>: [John]: Yeah, Maria, let's keep each other and everyone else motivated to make a difference! Together, our impact will surely last.
        # [Maria]: Yeah, John! Let's keep spreading kindness. It's awesome to know we can bring joy and comfort to those who need it.
        #     evidence = entry["qa_with_evidence"]
        last_turn = latest_dialogue[-1]
        last_turn_lines = last_turn.split("\n")
        for line in last_turn_lines:
            if not line.strip():
                continue
            # Phân tích người nói
            if line.startswith("["):
                new_item["conversations"].append({
                    "from": "system",
                    "value": "end",
                    "turn-info": f"system-turn-{str(turn_idx)}"
                })
                turn_idx += 1
    
    evidence = entry["qa_with_evidence"]
    for qa in evidence:
        question = qa["question"]
        answer = qa["answer"]
        if question:
            new_item["conversations"].append({
                "from": "human",
                "value": question,
                "turn-info": f"human-turn-{str(turn_idx)}"
            })
            turn_idx += 1
        if answer:
            
            evidence_texts = qa.get("evidence_texts", [])
            # chuyển "[" và "]" xuất hiện đầu tiên 1 lần duy nhất thành "" trong evidence_texts
            for i in range(len(evidence_texts)):
                evidence_texts[i] = evidence_texts[i].replace("[", "", 1).replace("]", "", 1)
            new_item["conversations"].append({
                "from": "chatbot",
                "value": answer,
                "turn-info": f"chatbot-turn-{str(turn_idx)}",
                "evidence_texts": evidence_texts
            })
            turn_idx += 1

    output.append(new_item)

# Ghi ra file
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)
