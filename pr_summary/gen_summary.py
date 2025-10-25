import argparse
import os
import json
import torch
from unsloth import FastLanguageModel

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
        'commits': []  # Store commits with their messages and file changes together
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


def create_title_prompt(pr_data):
    """
    Create a prompt for LLM to generate PR title.
    """
    prompt = "You are a helpful assistant that generates concise and descriptive pull request titles.\n\n"
    prompt += f"Repository: {pr_data['repository_name']}\n"
    prompt += f"Base Branch: {pr_data['base_branch']}\n"
    prompt += f"Head Branch: {pr_data['head_branch']}\n\n"
    
    # Add commit level details
    if pr_data['commits']:
        prompt += f"Commits ({len(pr_data['commits'])} commits):\n\n"
        for i, commit in enumerate(pr_data['commits'], 1):
            prompt += f"Commit {i}:\n"
            prompt += f"Message: {commit['message']}\n"
            
            # if commit['files']:
            #     prompt += f"Files Changed ({len(commit['files'])} files):\n"
            #     for file_change in commit['files']:
            #         prompt += f"\n  File: {file_change['filename']}\n"
            #         prompt += f"  Changes:\n"
            #         # Indent the patch for better readability
            #         for line in file_change['patch'].split('\n'):
            #             prompt += f"    {line}\n"
            prompt += "\n"
    
    # Add PR-level file changes
    if pr_data['file_changes']:
        prompt += f"PR-Level Files Changed ({len(pr_data['file_changes'])} files):\n"
        for file_change in pr_data['file_changes']:
            prompt += f"\nFile: {file_change['filename']}\n"
            prompt += f"Changes:\n{file_change['patch']}\n"
        prompt += "\n"
    
    prompt += "Based on the above information, generate a concise and descriptive pull request title (one line only):\n"
    prompt += "Title: "
    
    return prompt


def create_description_prompt(pr_data):
    """
    Create a prompt for LLM to generate PR description.
    """
    prompt = "You are a helpful assistant that generates comprehensive pull request descriptions.\n\n"
    prompt += f"Repository: {pr_data['repository_name']}\n"
    prompt += f"Base Branch: {pr_data['base_branch']}\n"
    prompt += f"Head Branch: {pr_data['head_branch']}\n\n"
    
    # Add commit level details
    if pr_data['commits']:
        prompt += f"Commits ({len(pr_data['commits'])} commits):\n\n"
        for i, commit in enumerate(pr_data['commits'], 1):
            prompt += f"Commit {i}:\n"
            prompt += f"Message: {commit['message']}\n"
            
            # if commit['files']:
            #     prompt += f"Files Changed ({len(commit['files'])} files):\n"
            #     for file_change in commit['files']:
            #         prompt += f"\n  File: {file_change['filename']}\n"
            #         prompt += f"  Changes:\n"
            #         # Indent the patch for better readability
            #         for line in file_change['patch'].split('\n'):
            #             prompt += f"    {line}\n"
            prompt += "\n"
    
    # Add PR-level file changes
    if pr_data['file_changes']:
        prompt += f"PR-Level Files Changed ({len(pr_data['file_changes'])} files):\n"
        for file_change in pr_data['file_changes']:
            prompt += f"\nFile: {file_change['filename']}\n"
            prompt += f"Changes:\n{file_change['patch']}\n"
        prompt += "\n"
    
    prompt += "Based on the above information, generate a comprehensive pull request description that explains:\n"
    prompt += "1. What changes were made\n"
    prompt += "2. Why these changes were made\n"
    prompt += "3. Impact of the changes\n\n"
    prompt += "Description:\n"
    
    return prompt


def generate_with_llm(model, tokenizer, prompt, max_new_tokens=150, temperature=0.3):
    """
    Generate text using the LLM model.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode the output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the input prompt)
    generated_text = full_output[len(prompt):].strip()
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate PR titles and descriptions using LLM")
    
    # Required parameters
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name or path (e.g., unsloth/mistral-7b-bnb-4bit)")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the input JSONL file containing PR data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated titles and descriptions")
    
    # Optional parameters
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for the model")
    parser.add_argument("--load_in_4bit", type=bool, default=True,
                        help="Load model in 4-bit quantization")
    parser.add_argument("--max_records", type=int, default=None,
                        help="Maximum number of records to process (None for all)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Unsloth model
    print(f"Loading model: {args.model_name}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    
    # Enable faster inference
    FastLanguageModel.for_inference(model)
    print("Model loaded successfully!", flush=True)
    
    # Read PR data from JSONL file
    print(f"Reading data from: {args.data_file}", flush=True)
    pr_records = []
    with open(args.data_file, 'r', encoding='utf-8') as f:
        for line in f:
            pr_records.append(json.loads(line))
    
    total_records = len(pr_records)
    if args.max_records:
        pr_records = pr_records[:args.max_records]
    
    print(f"Processing {len(pr_records)} out of {total_records} records", flush=True)
    
    # Output files - JSON format
    generated_output_file = os.path.join(args.output_dir, "generated_outputs.json")
    reference_output_file = os.path.join(args.output_dir, "reference_outputs.json")
    prompts_output_file = os.path.join(args.output_dir, "prompts.json")
    
    # Lists to store all outputs
    generated_outputs = []
    reference_outputs = []
    prompts_data = []
    
    for idx, pr_record in enumerate(pr_records):
        print(f"\n{'='*80}")
        print(f"Processing PR {idx + 1}/{len(pr_records)}")
        print(f"Repository: {pr_record.get('repository', 'N/A')}")
        print(f"PR Number: {pr_record.get('number', 'N/A')}")
        print(f"{'='*80}")
        
        try:
            # Collect PR data
            pr_data = collect_pr_data(pr_record)
            
            # Generate PR title
            print("Generating PR title...", flush=True)
            title_prompt = create_title_prompt(pr_data)
            generated_title = generate_with_llm(model, tokenizer, title_prompt, 
                                               max_new_tokens=50, temperature=0.3)
            # Clean up the title (take only the first line)
            generated_title = generated_title.split('\n')[0].strip()
            print(f"Generated Title: {generated_title}")
            
            # Generate PR description
            print("Generating PR description...", flush=True)
            description_prompt = create_description_prompt(pr_data)
            generated_description = generate_with_llm(model, tokenizer, description_prompt,
                                                     max_new_tokens=300, temperature=0.3)
            print(f"Generated Description: {generated_description[:200]}...")
            
            # Get reference title and description
            reference_title = pr_record.get('title', '')
            reference_description = pr_record.get('body', '')
            
            # Store generated outputs
            generated_outputs.append({
                'index': idx,
                'repository': pr_record.get('repository', ''),
                'pr_number': pr_record.get('number', ''),
                'base_branch': pr_record.get('base_branch', ''),
                'head_branch': pr_record.get('head_branch', ''),
                'generated_title': generated_title,
                'generated_description': generated_description
            })
            
            # Store reference outputs
            reference_outputs.append({
                'index': idx,
                'repository': pr_record.get('repository', ''),
                'pr_number': pr_record.get('number', ''),
                'base_branch': pr_record.get('base_branch', ''),
                'head_branch': pr_record.get('head_branch', ''),
                'reference_title': reference_title,
                'reference_description': reference_description
            })
            
            # Store prompts
            prompts_data.append({
                'index': idx,
                'repository': pr_record.get('repository', ''),
                'pr_number': pr_record.get('number', ''),
                'title_prompt': title_prompt,
                'description_prompt': description_prompt
            })
            
            print(f"✓ Successfully processed PR {idx + 1}")
            
        except Exception as e:
            print(f"✗ Error processing PR {idx + 1}: {str(e)}")
            # Store error information
            generated_outputs.append({
                'index': idx,
                'repository': pr_record.get('repository', ''),
                'pr_number': pr_record.get('number', ''),
                'error': str(e),
                'generated_title': '',
                'generated_description': ''
            })
            
            reference_outputs.append({
                'index': idx,
                'repository': pr_record.get('repository', ''),
                'pr_number': pr_record.get('number', ''),
                'reference_title': pr_record.get('title', ''),
                'reference_description': pr_record.get('body', '')
            })
            
            prompts_data.append({
                'index': idx,
                'repository': pr_record.get('repository', ''),
                'pr_number': pr_record.get('number', ''),
                'error': str(e),
                'title_prompt': '',
                'description_prompt': ''
            })
            continue
    
    # Save all outputs to JSON files
    print(f"\nSaving outputs to JSON files...", flush=True)
    
    with open(generated_output_file, 'w', encoding='utf-8') as f:
        json.dump(generated_outputs, f, indent=2, ensure_ascii=False)
    
    with open(reference_output_file, 'w', encoding='utf-8') as f:
        json.dump(reference_outputs, f, indent=2, ensure_ascii=False)
    
    with open(prompts_output_file, 'w', encoding='utf-8') as f:
        json.dump(prompts_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("Processing complete!")
    print(f"{'='*80}")
    print(f"Generated outputs saved to: {generated_output_file}")
    print(f"Reference outputs saved to: {reference_output_file}")
    print(f"Prompts saved to: {prompts_output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
