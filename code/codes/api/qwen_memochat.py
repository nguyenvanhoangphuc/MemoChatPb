import re
import json
import sys
import time
from random import sample
from tqdm import tqdm

# --- Thay openai bằng OpenAI‑compatible client của Qwen ---
from openai import OpenAI

# --- Thay tiktoken bằng transformers tokenizer ---
from transformers import AutoTokenizer, GPT2TokenizerFast

# === Cấu hình ===
input_data      = sys.argv[1]
qwen_model_path = sys.argv[2]        # ex: "/home/trungquang/LLM_models/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
api_key         = sys.argv[3]
output_path     = sys.argv[4]
prompt_path     = sys.argv[5]

# Khởi tạo client Qwen
client = OpenAI(
    api_key=api_key,
    base_url="http://localhost:8881/v1",
)

# Khởi tạo tokenizer của Qwen
try:
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_path, trust_remote_code=True)
except ValueError:
    print("⚠️ Không tìm thấy tokenizer Qwen; đang fallback sang GPT2TokenizerFast", file=sys.stderr)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Tham số chung
q_pre = ""
qa_link = ""
MaxLen = 2048
TarLen = 512
TaskTarLen = {
    "chatting_dialogsum": MaxLen,
    "chatting_alpacagpt4": MaxLen,
    "writing_topiocqa":   TarLen // 2,
    "writing_dialogsum":  TarLen,
    "retrieval_dialogsum":    32,
    "retrieval_topiocqa":      32
}

prompts = json.load(open(prompt_path, "r"))


def normalize_model_outputs(model_text):
    extracted = [re.sub(r'\s+', ' ', mt.replace('"','').replace("'", ""))
                 for mt in re.findall(r"'[^']*'|\"[^\"]*\"|\d+", model_text)]
    outs = []
    ti = 0
    while ti + 7 < len(extracted):
        if (extracted[ti]=="topic" and extracted[ti+2]=="summary" 
            and extracted[ti+4]=="start" and extracted[ti+6]=="end"):
            try:
                outs.append({
                    "topic":   extracted[ti+1],
                    "summary": extracted[ti+3],
                    "start":   int(extracted[ti+5]),
                    "end":     int(extracted[ti+7]),
                })
            except:
                pass
        ti += 1
    return outs

def normalize_chatting_outputs(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(' '.join(ln.split()) for ln in lines)

def gen_model_output(input_qs: str, task_type: str) -> str:
    # --- tính token bằng tokenizer của HuggingFace ---
    token_count = len(tokenizer.encode(input_qs))
    word_count  = len(input_qs.split())
    ratio = word_count / max(1, token_count)
    max_words = int((MaxLen - TarLen) * ratio)
    # chỉ lấy cuối để tránh quá dài
    print("max_words", max_words)
    input_qs = " ".join(input_qs.split()[-max_words:])
    print("input_qs", input_qs)
    target_len = TaskTarLen[task_type]
    messages = [
        {"role": "system",  "content": input_qs}
    ]
    
    # gọi API Qwen
    for _ in range(5):
        try:
            resp = client.chat.completions.create(
                model=qwen_model_path,
                messages=messages,
                max_tokens=target_len,
                temperature=0.2,
            )
            break
        except Exception:
            time.sleep(5)
    return resp.choices[0].message.content

def run_summary(history, memo, bot_thinking):
    system_insturction = prompts["writing_dialogsum"]["system"]
    task_instruction = prompts["writing_dialogsum"]["instruction"]
    history_log = "\n\n```\nTask Conversation:\n" + "\n".join(["(line {}) {}".format(h_i + 1, h.replace("\n", " ")) for h_i, h in enumerate(history["Recent Dialogs"][2:])])
    qs = q_pre + system_insturction.replace("LINE", str(len(history["Recent Dialogs"]) - 2)) + history_log + "\n```" + task_instruction.replace("LINE", str(len(history["Recent Dialogs"]) - 2)) + qa_link
    # print("-" * 20 + "summarizing" + "-" * 20)
    # print(qs)
    # print("-" * 20 + "summarizing" + "-" * 20)
    sum_history = gen_model_output(qs, "writing_dialogsum")
    sum_history = normalize_model_outputs(sum_history)
    # print("-" * 20 + "summarization" + "-" * 20)
    # print(sum_history)
    # print("-" * 20 + "summarization" + "-" * 20)
    for s in sum_history:
        memo[s["topic"]] = memo.get(s["topic"], []) + [{"summary": s["summary"], "dialogs": history["Recent Dialogs"][2:][(s["start"] - 1):s["end"]]}]
    if len(sum_history) == 0:
        si_0, si_1 = sample(list(range(len(history["Recent Dialogs"][2:]))), 2)
        memo["NOTO"].append({"summary": "Partial dialogs about: {} or {}.".format(history["Recent Dialogs"][2:][si_0], history["Recent Dialogs"][2:][si_1]), "dialogs": history["Recent Dialogs"][2:]})
    history["Recent Dialogs"] = history["Recent Dialogs"][-2:]
    bot_thinking["summarization"] = {"input": qs, "output": sum_history}
    return history, memo, bot_thinking

def run_retrieval(history, memo, bot_thinking):
    topics = []
    for k, v in memo.items():
        for vv in v:
            topics.append((k, vv["summary"], vv["dialogs"]))
    system_insturction = prompts["retrieval"]["system"]
    task_instruction = prompts["retrieval"]["instruction"]
    # task_case = "```\nQuery Sentence:\n" + history["User Input"][6:] + "\nTopic Options:\n" + \
    #             "\n".join(["({}) {}".format(v_i + 1, v[0] + ". " + v[1]) for v_i, v in enumerate(topics)]) + "\n```"
    # qs = q_pre + system_insturction.replace("OPTION", str(len(topics))) + task_case + task_instruction.replace("OPTION", str(len(topics))) + qa_link
    # print("-" * 20 + "retrieving" + "-" * 20)
    # print(qs)
    # print("-" * 20 + "retrieving" + "-" * 20)
    # outputs = gen_model_output(qs, "retrieval_dialogsum")
    # build list of "(i) topic. summary" strings
    options = ["({}) {}".format(i+1, t[0] + ". " + t[1]) for i, t in enumerate(topics)]

    qs = ""
    # split into sub‑batches if too many options
    sub_outputs = []
    max_opts_per_batch = 50  # hoặc tuỳ chỉnh cho phù hợp max token
    for i in range(0, len(options), max_opts_per_batch):
        sub_opts = options[i:i+max_opts_per_batch]
        sub_task_case = (
            "```\nQuery Sentence:\n" + history["User Input"][6:] +
            "\nTopic Options:\n" + "\n".join(sub_opts) + "\n```"
        )
        sub_qs = (
            q_pre +
            system_insturction.replace("OPTION", str(len(topics))) +
            sub_task_case +
            task_instruction.replace("OPTION", str(len(topics))) +
            qa_link
        )
        qs += sub_qs + "\n\n"  # ghép các sub_qs lại với nhau
        # gọi API cho từng sub_qs
        sub_out = gen_model_output(sub_qs, "retrieval_dialogsum")
        # mỗi sub_out dạng "89#90#91" v.v.
        sub_outputs.append(sub_out)
    # ghép lại thành một chuỗi duy nhất "89#90#91#104#105"
    outputs = "#".join([part.strip("#") for part in sub_outputs])

    print("-" * 20 + "retrieval" + "-" * 20)
    print(outputs)
    print("-" * 20 + "retrieval" + "-" * 20)
    outputs = outputs.split("#")
    chosen_topics = []
    for output in outputs:
        try:
            index_ = int(output) - 1
        except:
            continue
        if index_ < len(topics) and "NOTO" not in topics[index_]:
            chosen_topics.append(topics[index_])
    if len(chosen_topics) > 0:
        history["Related Topics"] = [ct[0] for ct in chosen_topics]
        history["Related Summaries"] = [ct[1] for ct in chosen_topics]
        history["Related Dialogs"] = [" ### ".join(ct[2]) for ct in chosen_topics]
    else:
        history["Related Topics"] = []
        history["Related Summaries"] = []
        history["Related Dialogs"] = []
    bot_thinking["retrieval"] = {"input": qs, "output": outputs}
    return history, bot_thinking

def run_eval():
    data = json.load(open(input_data, "r"))
    output_data = []
    
    # Nếu đã từng chạy 1 phần và output_path tồn tại, load lại để tiếp tục
    try:
        existing = json.load(open(output_path, "r"))
        # giả sử existing là list, ta khởi động lại từ len(existing)
        output_data = existing
        start_idx = len(existing)
        print(f"🔄 Resume from index {start_idx}")
    except FileNotFoundError:
        start_idx = 0
        print("🚀 Start fresh")
    
    for idx in tqdm(range(start_idx, len(data)), desc="Processing questions"):
        d = data[idx]
        print("=" * 20 + f" start of question {d['id']} " + "=" * 20)
        new_d = d.copy()  # hoặc deepcopy nếu cần

        history = {
            "Recent Dialogs": ["user: Hi!", "bot: Hi! How can I help you today?"], 
            "Related Topics": [], 
            "Related Summaries": [], 
            "Related Dialogs": [], 
            "User Input": "",
        }
        memo = {
            "NOTO": [{"summary": "None of the others.", "dialogs": []}]
        }

        for l_i in range(len(new_d["conversations"])):
            if l_i % 2 == 1:
                bot_thinking = {"retrieval": "", "summarization": ""}
                print("=" * 20 + "start of turn {}".format(l_i // 2 + 1) + "=" * 20)
                user = new_d["conversations"][l_i - 1]["from"] + ": " + new_d["conversations"][l_i - 1]["value"]
                mes_from = new_d["conversations"][l_i - 1]["from"]
                print(user + "\n\n")

                # create summary if recent dialogs exceed threshold
                if len(" ### ".join(history["Recent Dialogs"]).split(" ")) > (MaxLen // 2) or len(history["Recent Dialogs"]) >= 10:
                    print("Creating summary for recent dialogs...")
                    history, memo, bot_thinking = run_summary(history, memo, bot_thinking)
                
                # Sau đoạn này thì history["User Input"] sẽ là câu hỏi của người dùng hoặc là 1 câu trong đoạn hội thoại
                # Cần xác định câu hỏi của người dùng thì mới thực hiện retrieval và generate bot response
                if mes_from != "human":
                    bot_true = new_d["conversations"][l_i]["from"] + ": " + new_d["conversations"][l_i]["value"]
                    # nếu người dùng không nói gì thì bỏ qua
                    print("Bot: " + user + " is not a valid user input, skip this turn.\n")
                    history["Recent Dialogs"] += [user, bot_true]
                    continue

                # retrieve most related topics for every new user input
                history["User Input"] = user
                if len(memo.keys()) > 1:
                    history, bot_thinking = run_retrieval(history, memo, bot_thinking)
                
                # generate bot response
                system_insturction = prompts["chatting"]["system"]
                task_instruction = prompts["chatting"]["instruction"]
                task_case = "```\nRelated Evidences:\n" + "\n".join(["({}) {}".format(r_tsd_i + 1, {
                                "Related Topics": history["Related Topics"][r_tsd_i], 
                                "Related Summaries": history["Related Summaries"][r_tsd_i], 
                                "Related Dialogs": history["Related Dialogs"][r_tsd_i]
                            }) for r_tsd_i in range(len(history["Related Topics"]))]) + "\n\nRecent Dialogs:\n" + \
                            " ### ".join([hrd.replace("\n", " ") for hrd in history["Recent Dialogs"]]) + "\n```\n\nUser Input:\n" + history["User Input"] + " ### bot: "
                qs = q_pre + system_insturction + task_case + task_instruction + qa_link
                print("-" * 20 + "chatting" + "-" * 20)
                print(qs)
                print("-" * 20 + "chatting" + "-" * 20)
                outputs = gen_model_output(qs, "chatting_dialogsum")
                outputs = normalize_chatting_outputs(outputs)
                print("-" * 20 + "chatting output" + "-" * 20)
                print(outputs)
                print("-" * 20 + "chatting output" + "-" * 20)
                # history["Recent Dialogs"] += [user, "bot: " + outputs]
                print("bot: " + outputs + "\n")
                print("=" * 20 + "end of turn {}".format(l_i // 2 + 1) + "=" * 20)
                # print("\n\n\n\n")
                bot_thinking["answer"] = {"related_dialogs": [history["Related Dialogs"][r_tsd_i]
                            for r_tsd_i in range(len(history["Related Topics"]))],
                            "recent_dialogs": history["Recent Dialogs"]}
                # new_d["conversations"][l_i]["thinking"] = json.dumps(bot_thinking, ensure_ascii=False)
                new_d["conversations"][l_i]["thinking"] = bot_thinking
                new_d["conversations"][l_i]["value"] = outputs
                # sys.exit(0)  # Dừng lại để kiểm tra outputs

        # Sau khi hoàn thiện new_d, thêm vào output_data
        output_data.append(new_d)
        
        # 📝 Checkpoint: ghi luôn file output_path
        with open(output_path, "w") as fout:
            json.dump(output_data, fout, indent=2)
        
        print(f"✅ Finished question {d['id']} (index {idx}) – checkpointed.")
    
    print("🎉 All done!")

if __name__ == "__main__":
    run_eval()
