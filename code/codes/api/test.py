from openai import OpenAI

# Cấu hình API
api_key = "EMPTY"
api_url = "http://localhost:8881/v1"

client = OpenAI(
    api_key=api_key,
    base_url=api_url,
)

# Mẫu prompt cố định cho đánh giá relevance
SYSTEM_PROMPT = """<|im_start|>system
You are an evaluator assessing whether a given context is directly relevant to answering a specific question.
Evaluation Criteria:
Answer "yes" if:
    The context explicitly contains the information needed to answer the question.
    There is a direct and unambiguous match between the context content and the question’s requirements.
Answer "no" if:
    The context is too general, vague, or off-topic to support answering the question.
    The relationship between the context and the question is indirect, inferred, or ambiguous.
Key Focus:
    "yes" requires clear, explicit alignment—no guessing, no assumptions.
    "no" applies if there's any doubt or if the connection is only partial or suggestive.
Definition of "is relevant":
    "yes": The context provides direct evidence or information needed to answer the question.
    "no": The context does not directly help in answering the question.
<|im_end|>"""

def evaluate_relevance(context: str, question: str) -> str:
    prompt = SYSTEM_PROMPT + f"""\n<|im_start|>user
context: [{context}].
Question: [{question}]
<|im_end|>\n<|im_start|>assistant\n"""
    
    completion = client.completions.create(
        model="/home/trungquang/LLM_models/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        prompt=prompt,
        stream=False,
        max_tokens=50,
        temperature=0.0,
    )
    
    response_text = completion.choices[0].text.strip()
    return response_text


if __name__ == "__main__":
    context = "Albert Einstein was born in Germany and is known for the theory of relativity."
    question = "Where was Albert Einstein born?"

    result = evaluate_relevance(context, question)
    print(result)