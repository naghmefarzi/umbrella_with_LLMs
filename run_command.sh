# python src/main.py \
#   --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
#   --test_qrel_path "./data/dl2023/llm4eval_test_qrel_2024_withRel.txt" \
#   --queries_path "./data/dl2023/llm4eval_query_2024.txt" \
#   --docs_path "./data/dl2023/llm4eval_document_2024.jsonl" \
#   --prompt_mode "fewshot_bing" \
#   --result_file_path "./results/dl23_test_fewshot_bing_Llama-3-8B-Instruct.txt" \


# python src/main.py \
#   --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
#   --test_qrel_path "./data/dl2019/2019qrels-pass.txt" \
#   --queries_path "./data/dl2019/msmarco-test2019-queries.tsv" \
#   --docs_path "./data/dl2019/dl2019_document.jsonl" \
#   --prompt_mode "zeroshot_basic" \
#   --result_file_path "./results/dl19_test_zeroshot_basic_Llama-3-8B-Instruct.txt" \
#   # --max_pair 1


# python src/main.py \
#   --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
#   --test_qrel_path "./data/dl2020/2020qrels-pass.txt" \
#   --queries_path "./data/dl2020/msmarco-test2020-queries.tsv" \
#   --docs_path "./data/dl2020/dl2020_document.jsonl" \
#   --prompt_mode "zeroshot_basic" \
#   --result_file_path "./results/dl20_test_zeroshot_basic_Llama-3-8B-Instruct.txt" \

#----------------------------------------------------------------------------------------


python src/main.py \
  --model_id "google/flan-t5-large" \
  --test_qrel_path "./data/dl2023/llm4eval_test_qrel_2024_withRel.txt" \
  --queries_path "./data/dl2023/llm4eval_query_2024.txt" \
  --docs_path "./data/dl2023/llm4eval_document_2024.jsonl" \
  --prompt_mode "zeroshot_bing" \
  --result_file_path "./results/dl23_test_zeroshot_bing_flan-t5-large.txt" \


python src/main.py \
  --model_id "google/flan-t5-large" \
  --test_qrel_path "./data/dl2019/2019qrels-pass.txt" \
  --queries_path "./data/dl2019/msmarco-test2019-queries.tsv" \
  --docs_path "./data/dl2019/dl2019_document.jsonl" \
  --prompt_mode "zeroshot_bing" \
  --result_file_path "./results/dl19_test_zeroshot_bing_flan-t5-large.txt" \


python src/main.py \
  --model_id "google/flan-t5-large" \
  --test_qrel_path "./data/dl2020/2020qrels-pass.txt" \
  --queries_path "./data/dl2020/msmarco-test2020-queries.tsv" \
  --docs_path "./data/dl2020/dl2020_document.jsonl" \
  --prompt_mode "zeroshot_bing" \
  --result_file_path "./results/dl20_test_zeroshot_bing_flan-t5-large.txt" \



#----------------------------------------------------------------------------------------
# python src/main.py \
#   --model_id "deepseek-ai/DeepSeek-V3" \
#   --test_qrel_path "./data/dl2023/llm4eval_test_qrel_2024_withRel.txt" \
#   --queries_path "./data/dl2023/llm4eval_query_2024.txt" \
#   --docs_path "./data/dl2023/llm4eval_document_2024.jsonl" \
#   --prompt_mode "zeroshot_bing" \
#   --result_file_path "./results/dl23_test_zeroshot_bing_DSV3.txt" \
#   -together

# python src/main.py \
#   --model_id "deepseek-ai/DeepSeek-V3" \
#   --test_qrel_path "./data/dl2019/2019qrels-pass.txt" \
#   --queries_path "./data/dl2019/msmarco-test2019-queries.tsv" \
#   --docs_path "./data/dl2019/dl2019_document.jsonl" \
#   --prompt_mode "zeroshot_bing" \
#   --result_file_path "./results/dl19_test_zeroshot_bing_DSV3.txt" \
#   -together


  # python src/main.py \
  # --model_id "deepseek-ai/DeepSeek-V3" \
  # --test_qrel_path "./data/dl2020/2020qrels-pass.txt" \
  # --queries_path "./data/dl2020/msmarco-test2020-queries.tsv" \
  # --docs_path "./data/dl2020/dl2020_document.jsonl" \
  # --prompt_mode "zeroshot_bing" \
  # --result_file_path "./results/dl20_test_zeroshot_bing_DSV3.txt" \
  # -together