# PR Title & Description Generation using LLM
## Environment Setup
```sh
$ conda create -n pr-llm python=3.10
$ conda activate pr-llm
$ pip install -r requirements.txt
```
## Pull Request Crawling
To collect up to 1000 most recent merged PRs from the tp 100 starred projects:
```sh
$ cd crawling
$ python fetch_github_pr_data.py --repos 100 --prs-per-repo 1000 --output-dir github_pr_dataset_v3
```
## PR Data Preprocessing
The following command performs preprocessing using the techniques implemented in preprocess_pr_data.py, excluding entries with non-ASCII characters and entries with empty PR titles or descriptions:
```sh
$ cd pr_summary
$ python preprocess_pr_data.py --data_file <jsonl-dataset-filepath> --output_dir <output-folder-path> --exclude_non_ascii --exclude_missing_critical
```

## PR Summary Generation via LLM:
Example command:
```sh
$ cd pr_summary
$ python gen_summary.py --model_name unsloth/codegemma-7b-it-bnb-4bit --data_file <jsonl-dataset-filepath> --output_dir <output-filepath> --max_seq_length 65536
```

## Fine Tuning
Example command:
```sh
$ cd pr_summary
python fine_tune.py --model_name unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit --data_file <jsonl-dataset-filepath> --output_dir <finetuned-model-output-folder> --max_seq_length 65536 --num_train_epochs 5
```
After fine-tuning, you can use the `lora_model` folder path inside the model output folder as the `model_name`.

## Metric Evaluation
After generating the PR summary, to evaluate the response
```sh
$ cd pr_summary
$ python -m metrics.bleu --output-dir <pr-summary-folder>
$ python -m metrics.rouge --output-dir <pr-summary-folder>
```
A `Json` file containing scores will be saved in the corresponding folder.
