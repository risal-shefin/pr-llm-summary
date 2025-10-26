import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def remove_html_comments(text: str) -> str:
    """Remove HTML comments from text."""
    return re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)


def remove_checklist_paragraphs(text: str) -> str:
    """Remove paragraphs starting with 'checklist' headline."""
    # Remove sections that start with ## Checklist or ### Checklist (case insensitive)
    text = re.sub(r'#+\s*checklist[:\s]*.*?(?=\n##|\n###|$)', '', text, flags=re.IGNORECASE | re.DOTALL)
    return text


def should_filter_sentence(sentence: str) -> bool:
    """
    Check if a sentence should be filtered based on criteria:
    1) url, 2) internal reference (#123), 3) signature (signed-off-by),
    4) emails, 5) @name, 6) markdown headlines (## why)
    """
    # Check for URLs
    if re.search(r'https?://|www\.', sentence):
        return True
    
    # Check for internal references like #123
    if re.search(r'#\d+', sentence):
        return True
    
    # Check for signatures (signed-off-by, co-authored-by, etc.)
    if re.search(r'signed-off-by|co-authored-by|authored-by', sentence, re.IGNORECASE):
        return True
    
    # Check for emails
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', sentence):
        return True
    
    # Check for @mentions
    if re.search(r'@\w+', sentence):
        return True
    
    # Check for markdown headlines
    if re.search(r'^#+\s+', sentence.strip()):
        return True
    
    return False


def preprocess_tokens(tokens: List[str]) -> List[str]:
    """
    Preprocess tokens:
    - Replace SHA1 hash digests (7+ hex chars) with "sha"
    - Replace version strings (e.g., 1.2.3) with "version"
    - Replace numbers with 0
    """
    processed_tokens = []
    
    for token in tokens:
        # Check for SHA1 hash (7 or more hexadecimal characters)
        if re.fullmatch(r'[0-9a-fA-F]{7,}', token):
            processed_tokens.append('sha')
        # Check for version strings (e.g., 1.2.3, v1.2.3)
        elif re.match(r'v?\d+\.\d+(\.\d+)*', token):
            processed_tokens.append('version')
        # Check for numbers
        elif re.match(r'^\d+$', token):
            processed_tokens.append('0')
        else:
            processed_tokens.append(token)
    
    return processed_tokens


def preprocess_text(text: str) -> str:
    """
    Apply full preprocessing pipeline to text:
    1. Remove HTML comments
    2. Remove checklist paragraphs
    3. Split into sentences and filter unwanted ones
    4. Tokenize and preprocess tokens
    5. Reconstruct text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Remove HTML comments
    text = remove_html_comments(text)
    
    # Step 2: Remove checklist paragraphs
    text = remove_checklist_paragraphs(text)
    
    # Step 3: Split into sentences using NLTK
    sentences = sent_tokenize(text)
    
    # Filter sentences based on criteria
    filtered_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and not should_filter_sentence(sentence):
            filtered_sentences.append(sentence)
    
    # If no sentences remain, return empty string
    if not filtered_sentences:
        return ""
    
    # Step 4: Tokenize and preprocess each sentence
    processed_sentences = []
    for sentence in filtered_sentences:
        tokens = word_tokenize(sentence)
        processed_tokens = preprocess_tokens(tokens)
        processed_sentence = ' '.join(processed_tokens)
        processed_sentences.append(processed_sentence)
    
    # Step 5: Reconstruct text
    return ' '.join(processed_sentences)


def clean_pr_entry(pr_entry: Dict) -> Dict:
    """Clean a single PR entry by preprocessing title and description."""
    cleaned_entry = pr_entry.copy()
    
    # Preprocess title fields
    if 'reference_title' in cleaned_entry:
        cleaned_entry['reference_title'] = preprocess_text(cleaned_entry['reference_title'])
    if 'generated_title' in cleaned_entry:
        cleaned_entry['generated_title'] = preprocess_text(cleaned_entry['generated_title'])
    
    # Preprocess description fields
    if 'reference_description' in cleaned_entry:
        cleaned_entry['reference_description'] = preprocess_text(cleaned_entry['reference_description'])
    if 'generated_description' in cleaned_entry:
        cleaned_entry['generated_description'] = preprocess_text(cleaned_entry['generated_description'])
    
    return cleaned_entry


def is_valid_pr_entry(pr_entry: Dict) -> bool:
    """
    Check if a PR entry is valid (has non-empty title and description).
    """
    title_empty = True
    description_empty = True
    
    # Check title
    if 'reference_title' in pr_entry and pr_entry['reference_title'].strip():
        title_empty = False
    if 'generated_title' in pr_entry and pr_entry['generated_title'].strip():
        title_empty = False
    
    # Check description
    if 'reference_description' in pr_entry and pr_entry['reference_description'].strip():
        description_empty = False
    if 'generated_description' in pr_entry and pr_entry['generated_description'].strip():
        description_empty = False
    
    return not title_empty and not description_empty


def process_json_files(output_dir: Path, clean_output_dir: Path) -> Dict:
    """
    Process reference_outputs.json and generated_outputs.json files.
    Returns metadata about the processing.
    """
    metadata = {
        'original_count': 0,
        'filtered_count': 0,
        'removed_count': 0
    }
    
    reference_file = output_dir / 'reference_outputs.json'
    generated_file = output_dir / 'generated_outputs.json'
    
    # Check if files exist
    if not reference_file.exists() or not generated_file.exists():
        print(f"Warning: Required JSON files not found in {output_dir}")
        return metadata
    
    # Load JSON files
    with open(reference_file, 'r', encoding='utf-8') as f:
        reference_data = json.load(f)
    
    with open(generated_file, 'r', encoding='utf-8') as f:
        generated_data = json.load(f)
    
    metadata['original_count'] = len(reference_data)
    
    # Ensure both files have the same number of entries
    if len(reference_data) != len(generated_data):
        print(f"Warning: Mismatch in entry counts - reference: {len(reference_data)}, generated: {len(generated_data)}")
    
    # Process and filter entries
    cleaned_reference = []
    cleaned_generated = []
    
    for ref_entry, gen_entry in zip(reference_data, generated_data):
        # Clean both entries
        cleaned_ref = clean_pr_entry(ref_entry)
        cleaned_gen = clean_pr_entry(gen_entry)
        
        # Check if either entry is valid (has non-empty content)
        if is_valid_pr_entry(cleaned_ref) or is_valid_pr_entry(cleaned_gen):
            cleaned_reference.append(cleaned_ref)
            cleaned_generated.append(cleaned_gen)
        else:
            metadata['removed_count'] += 1
    
    metadata['filtered_count'] = len(cleaned_reference)
    
    # Save cleaned files
    with open(clean_output_dir / 'reference_outputs.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_reference, f, indent=2, ensure_ascii=False)
    
    with open(clean_output_dir / 'generated_outputs.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_generated, f, indent=2, ensure_ascii=False)
    
    return metadata


def save_metadata(clean_output_dir: Path, metadata: Dict):
    """Save metadata about the cleaning process."""
    metadata_file = clean_output_dir / 'cleaning_metadata.json'
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nCleaning Summary:")
    print(f"  Original PRs: {metadata['original_count']}")
    print(f"  Filtered PRs: {metadata['filtered_count']}")
    print(f"  Removed PRs: {metadata['removed_count']}")
    print(f"  Retention Rate: {metadata['filtered_count']/metadata['original_count']*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Clean and preprocess PR output files by removing templated and trivial information.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the directory containing reference_outputs.json and generated_outputs.json'
    )
    
    args = parser.parse_args()
    
    # Convert to Path object
    output_dir = Path(args.output_dir)
    
    # Validate output directory
    if not output_dir.exists():
        print(f"Error: Output directory '{output_dir}' does not exist.")
        return
    
    if not output_dir.is_dir():
        print(f"Error: '{output_dir}' is not a directory.")
        return
    
    # Create clean_outputs directory
    clean_output_dir = output_dir / 'clean_outputs'
    clean_output_dir.mkdir(exist_ok=True)
    
    print(f"Processing files in: {output_dir}")
    print(f"Output directory: {clean_output_dir}")
    
    # Process JSON files
    metadata = process_json_files(output_dir, clean_output_dir)
    
    # Save metadata
    save_metadata(clean_output_dir, metadata)
    
    print(f"\nCleaning completed successfully!")
    print(f"Cleaned files saved to: {clean_output_dir}")


if __name__ == '__main__':
    main()
