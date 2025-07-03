export GLOO_SOCKET_IFNAME=eth0
export WANDB_MODE=disabled

maindir=$1
datadir=${maindir}data
codedir=${maindir}code

qwen_settings=("memochat")

for qwen_setting in "${qwen_settings[@]}"
    do
    python3 ${codedir}/codes/api/qwen_${qwen_setting}.py \
        ${datadir}/locomo/memochat_locomo10_root.json \
        /home/trungquang/LLM_models/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 \
        EMPTY \
        ${datadir}/locomo/locomo_testing/locomo_temp_qwen-${qwen_setting}.json \
        ${datadir}/prompts.json
    done

# for qwen_setting in "${qwen_settings[@]}"
#     do
#     python3 ${codedir}/codes/api/qwen_${qwen_setting}.py \
#         ${datadir}/locomo/memochat_locomo10_root.json \
#         /home/trungquang/LLM_models/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 \
#         EMPTY \
#         ${datadir}/locomo/locomo_testing/locomo_testing_qwen-${qwen_setting}.json \
#         ${datadir}/prompts.json
#     done