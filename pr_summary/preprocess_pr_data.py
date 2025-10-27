import argparse
import os
import json
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


def remove_html_comments(text):
    """Remove HTML comments from text."""
    if not text:
        return text
    # Remove HTML comments <!-- ... -->
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    return text


def remove_checklist_paragraphs(text):
    """Remove paragraphs starting with 'checklist' headline."""
    if not text:
        return text
    # Remove sections with checklist headlines (case-insensitive)
    # Matches markdown headers like "## Checklist" or "### checklist" followed by content
    text = re.sub(r'(?i)^#{1,6}\s*checklist.*?(?=^#{1,6}\s|\Z)', '', text, flags=re.MULTILINE | re.DOTALL)
    # Also remove lines that start with "checklist:" or similar
    text = re.sub(r'(?i)^checklist:.*?(?=\n\n|\Z)', '', text, flags=re.MULTILINE | re.DOTALL)
    return text


def should_filter_sentence(sentence):
    """
    Check if a sentence should be filtered out based on the criteria.
    Returns True if sentence should be removed.
    """
    if not sentence:
        return True
    
    # 1) Check for URLs
    url_pattern = r'https?://|www\.'
    if re.search(url_pattern, sentence, re.IGNORECASE):
        return True
    
    # 2) Check for internal references like #123
    internal_ref_pattern = r'#\d+'
    if re.search(internal_ref_pattern, sentence):
        return True
    
    # 3) Check for signatures like "signed-off-by"
    signature_pattern = r'signed-off-by|co-authored-by|reviewed-by'
    if re.search(signature_pattern, sentence, re.IGNORECASE):
        return True
    
    # 4) Check for emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if re.search(email_pattern, sentence):
        return True
    
    # 5) Check for @mentions
    mention_pattern = r'@\w+'
    if re.search(mention_pattern, sentence):
        return True
    
    # 6) Check for markdown headlines
    headline_pattern = r'^#{1,6}\s+'
    if re.match(headline_pattern, sentence.strip()):
        return True
    
    return False


def replace_special_tokens(tokens):
    """
    Replace special tokens:
    - SHA1 hashes (7+ hex chars) -> "sha"
    - Version strings (e.g., "1.2.3") -> "version"
    - Numbers -> "0"
    """
    processed_tokens = []
    
    for token in tokens:
        # Check if token is SHA1 hash (7 or more hex characters)
        if re.match(r'^[0-9a-fA-F]{7,}$', token):
            processed_tokens.append('sha')
        # Check if token is a version string (e.g., 1.2.3, v1.2.3)
        elif re.match(r'^v?\d+\.\d+(\.\d+)*$', token):
            processed_tokens.append('version')
        # Check if token is a number
        elif re.match(r'^\d+\.?\d*$', token):
            processed_tokens.append('0')
        else:
            processed_tokens.append(token)
    
    return processed_tokens


def remove_non_ascii_tokens(tokens):
    """
    Remove tokens with non-ASCII characters.
    Returns tuple: (filtered_tokens, non_ascii_ratio)
    """
    ascii_tokens = []
    non_ascii_count = 0
    
    for token in tokens:
        try:
            # Check if token contains only ASCII characters
            token.encode('ascii')
            ascii_tokens.append(token)
        except UnicodeEncodeError:
            non_ascii_count += 1
    
    total_tokens = len(tokens)
    non_ascii_ratio = non_ascii_count / total_tokens if total_tokens > 0 else 0
    
    return ascii_tokens, non_ascii_ratio


def preprocess_text(text):
    """
    Apply all preprocessing steps to a text.
    Returns tuple: (processed_text, is_non_ascii)
    """
    if not text or not isinstance(text, str):
        return "", False
    
    # Step 1: Remove HTML comments
    text = remove_html_comments(text)
    
    # Step 2: Remove checklist paragraphs
    text = remove_checklist_paragraphs(text)
    
    # Step 3: Split into sentences and filter
    sentences = sent_tokenize(text)
    filtered_sentences = [s for s in sentences if not should_filter_sentence(s)]
    
    if not filtered_sentences:
        return "", False
    
    # Step 4: Tokenize and process tokens
    all_tokens = []
    for sentence in filtered_sentences:
        tokens = word_tokenize(sentence)
        all_tokens.extend(tokens)
    
    # Step 5: Replace special tokens
    all_tokens = replace_special_tokens(all_tokens)
    
    # Step 6: Remove non-ASCII tokens
    ascii_tokens, non_ascii_ratio = remove_non_ascii_tokens(all_tokens)
    
    # Check if text should be marked as non-ASCII
    is_non_ascii = non_ascii_ratio > 0.5
    
    # Reconstruct text from tokens
    processed_text = ' '.join(ascii_tokens)
    
    return processed_text, is_non_ascii


def collect_and_preprocess_pr_data(pr_record):
    """
    Extract and preprocess relevant data from a PR record.
    Returns a dictionary with preprocessed data and metadata.
    """
    processed_data = {
        'repository': pr_record.get('repository', ''),
        'number': pr_record.get('number', ''),
        'base_branch': pr_record.get('base_branch', ''),
        'head_branch': pr_record.get('head_branch', ''),
        'title': '',
        'description': '',
        'commit_messages': [],
        'file_changes': [],
        'is_non_ascii': False,
        'has_missing_critical_data': False
    }
    
    # Preprocess PR title
    title = pr_record.get('title', '')
    processed_title, title_non_ascii = preprocess_text(title)
    processed_data['title'] = processed_title
    
    # Preprocess PR description/body
    body = pr_record.get('body', '')
    processed_body, body_non_ascii = preprocess_text(body)
    processed_data['description'] = processed_body
    
    # Preprocess commit messages
    if 'commit_list' in pr_record:
        for commit in pr_record['commit_list']:
            if 'message' in commit and commit['message']:
                processed_msg, msg_non_ascii = preprocess_text(commit['message'])
                if processed_msg:  # Only add non-empty messages
                    processed_data['commit_messages'].append(processed_msg)
    
    # Process file changes (extract filename and patch)
    if 'files' in pr_record:
        for file_info in pr_record['files']:
            file_change = {
                'filename': file_info.get('filename', ''),
                'status': file_info.get('status', ''),
                'additions': file_info.get('additions', 0),
                'deletions': file_info.get('deletions', 0),
                'changes': file_info.get('changes', 0),
            }
            
            # Preprocess patch/diff if available
            if 'patch' in file_info and file_info['patch']:
                # Extract comments from patch (lines starting with # or //)
                patch = file_info['patch']
                comment_lines = []
                for line in patch.split('\n'):
                    # Extract single-line comments
                    if re.match(r'^\+\s*(#|//)', line):
                        comment = re.sub(r'^\+\s*(#|//)\s*', '', line)
                        comment_lines.append(comment)
                
                # Preprocess combined comments
                if comment_lines:
                    combined_comments = ' '.join(comment_lines)
                    processed_comments, _ = preprocess_text(combined_comments)
                    file_change['processed_comments'] = processed_comments
            
            processed_data['file_changes'].append(file_change)
    
    # Check if the entire PR should be marked as non-ASCII
    # Consider title, body, and commit messages
    is_non_ascii = title_non_ascii or body_non_ascii
    processed_data['is_non_ascii'] = is_non_ascii
    
    # Mark as having missing critical data if critical fields are empty after preprocessing
    if not processed_title or not processed_body or not processed_data['commit_messages']:
        processed_data['has_missing_critical_data'] = True
    
    return processed_data


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess PR data by filtering trivial and templated information"
    )
    
    # Required parameters
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the input JSONL file containing PR data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save preprocessed data")
    
    # Optional parameters
    parser.add_argument("--output_prefix", type=str, default="pr_dataset_preprocessed",
                        help="Prefix for output files")
    parser.add_argument("--exclude_non_ascii", action='store_true',
                        help="Exclude PRs marked as non-ASCII from output")
    parser.add_argument("--exclude_missing_critical", action='store_true',
                        help="Exclude PRs with missing critical fields after preprocessing")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read PR data from JSONL file
    print(f"Reading data from: {args.data_file}")
    print(f"{'='*80}")
    
    pr_records = []
    with open(args.data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pr_records.append(json.loads(line))
    
    print(f"Total records read: {len(pr_records)}")
    
    # Process each PR record
    processed_records = []
    stats = {
        'total': len(pr_records),
        'non_ascii': 0,
        'missing_critical_data': 0,
        'processed': 0
    }
    
    for idx, pr_record in enumerate(pr_records):
        if (idx + 1) % 100 == 0:
            print(f"Processing record {idx + 1}/{len(pr_records)}...")
        
        try:
            processed_pr = collect_and_preprocess_pr_data(pr_record)
            
            # Update statistics
            if processed_pr['is_non_ascii']:
                stats['non_ascii'] += 1
            if processed_pr['has_missing_critical_data']:
                stats['missing_critical_data'] += 1
            
            # Check exclusion criteria
            if args.exclude_non_ascii and processed_pr['is_non_ascii']:
                continue
            if args.exclude_missing_critical and processed_pr['has_missing_critical_data']:
                continue
            
            processed_records.append(processed_pr)
            stats['processed'] += 1
            
        except Exception as e:
            print(f"Error processing record {idx + 1}: {str(e)}")
            continue
    
    # Save preprocessed data
    output_json = os.path.join(args.output_dir, f"{args.output_prefix}.json")
    output_jsonl = os.path.join(args.output_dir, f"{args.output_prefix}.jsonl")
    
    print(f"\n{'='*80}")
    print("Saving preprocessed data...")
    
    # Save as JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(processed_records, f, indent=2, ensure_ascii=False)
    
    # Save as JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for record in processed_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # Print statistics
    print(f"{'='*80}")
    print("Preprocessing complete!")
    print(f"{'='*80}")
    print(f"Total records: {stats['total']}")
    print(f"Non-ASCII marked: {stats['non_ascii']} ({stats['non_ascii']/stats['total']*100:.2f}%)")
    print(f"Missing critical data: {stats['missing_critical_data']} ({stats['missing_critical_data']/stats['total']*100:.2f}%)")
    print(f"Records saved: {stats['processed']}")
    print(f"{'='*80}")
    print(f"Output files:")
    print(f"  JSON: {output_json}")
    print(f"  JSONL: {output_jsonl}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
