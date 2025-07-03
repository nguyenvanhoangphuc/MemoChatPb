## Chạy môi trường
conda activate phuc_new

## Cài thư viện
pip install -r core_requirement.txt
python -m pip install transformers
<!-- python -m pip install --upgrade "torch>=2.1" --index-url https://download.pytorch.org/whl/cu118 -->
python -m pip install --force-reinstall torch==2.2.0 torchvision==0.17.0 --index-url https://download.pyto
rch.org/whl/cu118

## Run
### Đánh giá trên tập locomo
- Model Qwen-MemoChat
bash code/scripts/memochat_qwen_new.sh /home/hoangphuc/MemoChat/ > output_qwen_new.txt
- MemoChat


### Chạy các metric
python code/codes/eval/evaluation_new.py --preds /home/hoangphuc/MemoChat/data/locomo/locomo_testing/locomo_testing_qwen-memochat.json --model microsoft/deberta-xlarge-mnli


python code/codes/eval/evaluation_new.py --preds /home/hoangphuc/MemoChat/data/locomo/locomo_testing/locomo_testing_t5-3b_1k.json --model microsoft/deberta-xlarge-mnli


python code/codes/eval/evaluation_full.py --preds /home/hoangphuc/MemoChat/data/locomo/locomo_testing/locomo_testing_qwen-memochat.json --model microsoft/deberta-xlarge-mnli

python code/codes/eval/evaluation_full.py --preds /home/hoangphuc/MemoChat/data/locomo/locomo_testing/locomo_testing_t5-3b_1k.json --model microsoft/deberta-xlarge-mnli

python code/codes/eval/evaluation_full.py --preds /home/hoangphuc/MemoChat/data/locomo/locomo_testing/locomo_temp_qwen-memochat.json --model microsoft/deberta-xlarge-mnli