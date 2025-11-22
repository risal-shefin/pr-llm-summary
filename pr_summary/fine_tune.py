import argparse
import os
import json
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset


def collect_pr_data(pr_record):
    """
    Extract relevant data from a PR record for LLM input.
    Returns a dictionary with collected data.
    """
    collected_data = {
        'repository_name': pr_record.get('repository', ''),
        'base_branch': pr_record.get('base_branch', ''),
        'head_branch': pr_record.get('head_branch', ''),
        'file_changes': [],
        'commits': []
    }
    
    # Collect file changes (patches) from files list
    if 'files' in pr_record:
        for file_info in pr_record['files']:
            if 'patch' in file_info and file_info['patch']:
                collected_data['file_changes'].append({
                    'filename': file_info.get('filename', ''),
                    'patch': file_info.get('patch', '')
                })
    
    # Collect commits with their messages and file changes grouped together
    if 'commit_list' in pr_record:
        for commit in pr_record['commit_list']:
            commit_data = {
                'message': commit.get('message', ''),
                'files': []
            }
            
            # Collect file changes for this commit
            if 'files' in commit:
                for file_info in commit['files']:
                    if 'patch' in file_info and file_info['patch']:
                        commit_data['files'].append({
                            'filename': file_info.get('filename', ''),
                            'patch': file_info.get('patch', '')
                        })
            
            collected_data['commits'].append(commit_data)
    
    return collected_data


def create_training_prompt(pr_data, title, description):
    """
    Create a training prompt in conversational format for fine-tuning.
    Uses instruction-response format suitable for chat models.
    """
    # Create the instruction part with PR context
    instruction = "You are a helpful assistant that generates pull request titles and descriptions.\n\n"
    instruction += f"Repository: {pr_data['repository_name']}\n"
    instruction += f"Base Branch: {pr_data['base_branch']}\n"
    instruction += f"Head Branch: {pr_data['head_branch']}\n\n"
    
    # Add commit level details
    if pr_data['commits']:
        instruction += f"Commits ({len(pr_data['commits'])} commits):\n\n"
        for i, commit in enumerate(pr_data['commits'], 1):
            instruction += f"Commit {i}:\n"
            instruction += f"Message: {commit['message']}\n\n"
    
    # Add PR-level file changes
    if pr_data['file_changes']:
        instruction += f"PR-Level Files Changed ({len(pr_data['file_changes'])} files):\n"
        for file_change in pr_data['file_changes']:
            instruction += f"\nFile: {file_change['filename']}\n"
            # Truncate very long patches to avoid token limits
            patch = file_change['patch']
            if len(patch) > 1000:
                patch = patch[:1000] + "\n... (truncated)"
            instruction += f"Changes:\n{patch}\n"
        instruction += "\n"
    
    instruction += "Based on the above information, generate a concise pull request title and a comprehensive description."
    
    # Create the expected response
    response = f"Title: {title}\n\nDescription:\n{description}"
    
    return instruction, response


def format_for_chat_template(tokenizer, instruction, response):
    """
    Format the instruction and response using the model's chat template.
    """
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response}
    ]
    
    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return formatted_text


def prepare_dataset(data_file, tokenizer, max_records=None, validation_split=0.1, seed=42):
    """
    Prepare the dataset for training from JSONL file.
    Returns a tuple of (train_dataset, eval_dataset).
    """
    print(f"Reading data from: {data_file}", flush=True)
    pr_records = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            pr_records.append(json.loads(line))
    
    if max_records:
        pr_records = pr_records[:max_records]
    
    print(f"Processing {len(pr_records)} records for training", flush=True)
    
    # Prepare training examples
    training_data = []
    
    for idx, pr_record in enumerate(pr_records):
        # Collect PR data
        pr_data = collect_pr_data(pr_record)
        
        # Get reference title and description
        title = pr_record.get('title', '')
        description = pr_record.get('description', '')
        
        # Skip if title or description is empty
        if not title or not description:
            print(f"Skipping record {idx}: missing title or description")
            continue
        
        # Create training prompt
        instruction, response = create_training_prompt(pr_data, title, description)
        
        # Format using chat template
        formatted_text = format_for_chat_template(tokenizer, instruction, response)
        
        training_data.append({
            'text': formatted_text,
            'repository': pr_record.get('repository', ''),
            'pr_number': pr_record.get('number', '')
        })
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(pr_records)} records", flush=True)
            
    
    print(f"Successfully prepared {len(training_data)} training examples", flush=True)
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(training_data)
    
    # Split into train and validation sets
    print(f"\nSplitting dataset: {int((1-validation_split)*100)}% train, {int(validation_split*100)}% validation", flush=True)
    split_dataset = dataset.train_test_split(test_size=validation_split, seed=seed, shuffle=True)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"Train set size: {len(train_dataset)} examples", flush=True)
    print(f"Validation set size: {len(eval_dataset)} examples", flush=True)
    
    return train_dataset, eval_dataset


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune LLM for PR title and description generation using Unsloth")
    
    # Required parameters
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name or path (e.g., unsloth/mistral-7b-bnb-4bit, unsloth/llama-3.1-8b-bnb-4bit)")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the input JSONL file containing PR data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned model")
    
    # Model parameters
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for the model")
    parser.add_argument("--load_in_4bit", type=bool, default=True,
                        help="Load model in 4-bit quantization")
    parser.add_argument("--max_records", type=int, default=None,
                        help="Maximum number of records to use for training (None for all)")
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension (rank)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha parameter for scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout probability")
    parser.add_argument("--target_modules", type=str, nargs='+', 
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="Target modules for LoRA")
    
    # Training parameters
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=5,
                        help="Number of warmup steps")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum number of training steps (-1 for full training)")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every X updates steps")
    
    # Optimizer parameters
    parser.add_argument("--optim", type=str, default="adamw_8bit",
                        help="Optimizer to use (adamw_8bit recommended for memory efficiency)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        help="Learning rate scheduler type")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--fp16", action="store_true",
                        help="Use fp16 training")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bf16 training")
    
    # Validation parameters
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="Fraction of data to use for validation (default: 0.1 for 10%)")
    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Evaluate on validation set every X steps")
    
    return parser.parse_args()


def print_configuration(args):
    """Print training configuration."""
    print(f"\n{'='*80}")
    print("FINE-TUNING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Data file: {args.data_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"LoRA rank (r): {args.lora_r}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Validation split: {int(args.validation_split*100)}%")
    print(f"Evaluation frequency: every {args.eval_steps} steps")
    print(f"{'='*80}\n")


def load_and_prepare_model(args):
    """Load model and tokenizer, then configure LoRA."""
    print("Loading model and tokenizer...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    print("Model loaded successfully!", flush=True)
    
    # Configure LoRA
    print("\nConfiguring LoRA...", flush=True)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=args.target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized gradient checkpointing
        random_state=args.seed,
        use_rslora=False,  # Rank-stabilized LoRA
        loftq_config=None,  # LoftQ quantization
    )
    print("LoRA configuration complete!", flush=True)
    
    return model, tokenizer


def create_training_arguments(args):
    """Create TrainingArguments for the trainer."""
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16 if not args.bf16 else False,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        
        # Evaluation strategy
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        
        # Save checkpoints and keep best model
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=5,  # Keep n most recent checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # TensorBoard logging
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "logs"),
        
        push_to_hub=False,
    )


def find_best_checkpoint(output_dir):
    """Find the checkpoint with the lowest training loss."""
    print("Finding best checkpoint...", flush=True)
    best_checkpoint = None
    best_loss = float('inf')
    
    for checkpoint_dir in os.listdir(output_dir):
        if checkpoint_dir.startswith("checkpoint-"):
            trainer_state_path = os.path.join(output_dir, checkpoint_dir, "trainer_state.json")
            if os.path.exists(trainer_state_path):
                with open(trainer_state_path, 'r') as f:
                    state = json.load(f)
                    # Get the loss from the last log entry for this checkpoint
                    if state.get('log_history'):
                        for entry in reversed(state['log_history']):
                            if 'loss' in entry:
                                loss = entry['loss']
                                if loss < best_loss:
                                    best_loss = loss
                                    best_checkpoint = os.path.join(output_dir, checkpoint_dir)
                                break
    
    return best_checkpoint, best_loss


def save_models(model, tokenizer, output_dir, args):
    """Save the fine-tuned model in various formats."""
    print("Saving the fine-tuned model...", flush=True)
    
    # Save LoRA adapters
    model.save_pretrained(os.path.join(output_dir, "lora_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "lora_model"))
    
    # # Save merged 16-bit model
    # print("\nSaving merged 16-bit model...", flush=True)
    # model.save_pretrained_merged(
    #     os.path.join(output_dir, "merged_16bit"),
    #     tokenizer,
    #     save_method="merged_16bit",
    # )
    
    # # Save merged 4-bit model
    # print("Saving merged 4-bit model...", flush=True)
    # model.save_pretrained_merged(
    #     os.path.join(output_dir, "merged_4bit"),
    #     tokenizer,
    #     save_method="merged_4bit",
    # )
    
    # # Save GGUF format
    # print("Saving GGUF format...", flush=True)
    # model.save_pretrained_gguf(
    #     os.path.join(output_dir, "gguf"),
    #     tokenizer,
    #     quantization_method="q4_k_m",
    # )
    
    # Save training configuration
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n{'='*80}")
    print("ALL MODELS SAVED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"LoRA adapters: {os.path.join(output_dir, 'lora_model')}")
    print(f"Merged 16-bit: {os.path.join(output_dir, 'merged_16bit')}")
    print(f"Merged 4-bit: {os.path.join(output_dir, 'merged_4bit')}")
    print(f"GGUF format: {os.path.join(output_dir, 'gguf')}")
    print(f"Training config: {config_path}")
    print(f"{'='*80}\n")


def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print_configuration(args)
    
    # Load model and prepare with LoRA
    model, tokenizer = load_and_prepare_model(args)
    
    # Prepare dataset with train/validation split
    print("\nPreparing dataset...", flush=True)
    train_dataset, eval_dataset = prepare_dataset(
        args.data_file, 
        tokenizer, 
        args.max_records,
        validation_split=args.validation_split,
        seed=args.seed
    )
    
    # Create training arguments
    training_args = create_training_arguments(args)
    
    # Initialize trainer
    print("\nInitializing trainer...", flush=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    
    # Start training
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")
    
    trainer.train()
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}\n")
    
    print("Best model automatically loaded (based on lowest validation loss)", flush=True)
    save_models(model, tokenizer, args.output_dir, args)


if __name__ == "__main__":
    main()
