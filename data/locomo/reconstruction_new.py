import json

# Mở và đọc file
with open('/home/hoangphuc/MemoChat/data/locomo/locomo10.json', 'r', encoding='utf-8') as f:
    loaded_data = json.load(f)
 
data = []
 
for row in loaded_data:
    conversation = row["conversation"]
    qa_list = row.get("qa", [])
 
    # Lấy tất cả session_* (bỏ session_*_date_time)
    sessions = {
        key: value
        for key, value in conversation.items()
        if key.startswith("session_") and not key.endswith("_date_time")
    }
 
    # Tạo mapping dia_id → {speaker, text} + thu thập speaker
    dia_id_to_text = {}
    speakers_set = set()
 
    for turns in sessions.values():
        for turn in turns:
            dia_id_to_text[turn["dia_id"]] = {
                "speaker": turn["speaker"],
                "text": turn["text"]
            }
            speakers_set.add(turn["speaker"])
 
    speakers = sorted(list(speakers_set))  # nếu cần sắp xếp
 
    # Tạo history: list các session, mỗi session là list cặp turn
    history = []
    for session_name, turns in sessions.items():
        session_data = []
        i = 0
        while i < len(turns) - 1:
            current_turn = turns[i]
            next_turn = turns[i + 1]
 
            # Chỉ ghép nếu hai speaker khác nhau
            if current_turn["speaker"] != next_turn["speaker"]:
                pair_text = (
                    f"<Turn {len(session_data)}> "
                    f"[{current_turn['speaker']}]: {current_turn['text']}\n"
                    f"[{next_turn['speaker']}]: {next_turn['text']}"
                )
                session_data.append(pair_text)
                i += 2
            else:
                # Nếu cùng speaker, bỏ qua 1 để tránh ghép sai
                i += 1
        history.append(session_data)    
    # Xử lý QA: thêm phần evidence text theo dia_id
    qa_with_evidence = []
    for qa_item in qa_list:
        if not qa_item.get("answer"):
            continue
 
        evidence_ids = qa_item.get("evidence", [])
        evidence_texts = []
        for eid in evidence_ids:
            if eid in dia_id_to_text:
                e = dia_id_to_text[eid]
                evidence_texts.append(f"[{e['speaker']}]: {e['text']}")
            else:
                evidence_texts.append(f"[{eid}]: <Not found>")
 
        qa_with_evidence.append({
            "question": qa_item.get("question", ""),
            "answer": qa_item.get("answer", ""),
            "evidence_dia_ids": evidence_ids,
            "evidence_texts": evidence_texts
        })
 
    if qa_with_evidence:
        data.append({
            "history": history,
            "qa_with_evidence": qa_with_evidence,
            "speakers": speakers
        })
    
# Lưu dữ liệu đã xử lý vào file JSON mới
output_file = '/home/hoangphuc/MemoChat/data/locomo/secom_locomo10_new.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)