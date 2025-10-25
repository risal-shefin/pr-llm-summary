#!/usr/bin/env python3
"""
Clean existing PR JSON files to remove HTML comments from body text.
Run this on already collected data to fix the HTML comment issue.
"""

import json
import re
from pathlib import Path
import argparse


def clean_body_text(text: str) -> str:
    """
    Remove HTML comments and clean up body text.
    
    Args:
        text: Raw body text that may contain HTML comments
        
    Returns:
        Cleaned body text
    """
    if not text:
        return ""
    
    # Remove HTML comments like <!-- ... -->
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def clean_json_file(file_path: Path) -> bool:
    """
    Clean a single JSON file by removing HTML comments from body.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        True if file was modified, False otherwise
    """
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if body needs cleaning
        original_body = data.get('body', '')
        
        if not original_body:
            return False
        
        # Clean the body
        cleaned_body = clean_body_text(original_body)
        
        # Check if anything changed
        if cleaned_body == original_body:
            return False
        
        # Update the data
        data['body'] = cleaned_body
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def clean_directory(directory: str, pattern: str = "*.json") -> tuple[int, int]:
    """
    Clean all JSON files in a directory.
    
    Args:
        directory: Directory containing JSON files
        pattern: File pattern to match (default: *.json)
        
    Returns:
        Tuple of (files_processed, files_modified)
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Directory not found: {directory}")
        return 0, 0
    
    # Find all matching files
    json_files = list(dir_path.glob(pattern))
    
    if not json_files:
        print(f"No files matching '{pattern}' found in {directory}")
        return 0, 0
    
    print(f"Found {len(json_files)} files to process...")
    
    processed = 0
    modified = 0
    
    for file_path in json_files:
        # Skip the aggregate files
        if file_path.name in ['pr_dataset_complete.json', 'pr_dataset_complete.jsonl']:
            continue
        
        processed += 1
        
        if clean_json_file(file_path):
            modified += 1
            print(f"  ✓ Cleaned: {file_path.name}")
        else:
            print(f"  - No changes: {file_path.name}")
    
    return processed, modified


def clean_aggregate_files(directory: str):
    """
    Rebuild aggregate JSON and JSONL files from cleaned individual files.
    
    Args:
        directory: Directory containing JSON files
    """
    dir_path = Path(directory)
    
    # Find all individual PR files
    pr_files = [f for f in dir_path.glob("*_pr_*.json") 
                if f.name not in ['pr_dataset_complete.json']]
    
    if not pr_files:
        print("No individual PR files found to aggregate")
        return
    
    print(f"\nRebuilding aggregate files from {len(pr_files)} PR files...")
    
    all_data = []
    
    for file_path in sorted(pr_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.append(data)
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
    
    # Save complete JSON
    json_file = dir_path / 'pr_dataset_complete.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: {json_file}")
    
    # Save JSONL
    jsonl_file = dir_path / 'pr_dataset_complete.jsonl'
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for pr in all_data:
            f.write(json.dumps(pr, ensure_ascii=False) + '\n')
    print(f"  ✓ Saved: {jsonl_file}")
    
    # Update CSV summary
    csv_file = dir_path / 'pr_dataset_summary.csv'
    import csv
    
    if all_data:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'repository', 'number', 'title', 'state', 'merged',
                'author', 'created_at', 'additions', 'deletions',
                'changed_files', 'commits', 'labels'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for pr in all_data:
                row = {
                    'repository': pr.get('repository', ''),
                    'number': pr.get('number', 0),
                    'title': pr.get('title', ''),
                    'state': pr.get('state', ''),
                    'merged': pr.get('merged', False),
                    'author': pr.get('author', {}).get('login', ''),
                    'created_at': pr.get('created_at', ''),
                    'additions': pr.get('additions', 0),
                    'deletions': pr.get('deletions', 0),
                    'changed_files': pr.get('changed_files', 0),
                    'commits': pr.get('commits', 0),
                    'labels': ', '.join(pr.get('labels', []))
                }
                writer.writerow(row)
        print(f"  ✓ Saved: {csv_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Clean HTML comments from PR JSON files'
    )
    parser.add_argument(
        '--directory',
        default='github_pr_dataset',
        help='Directory containing JSON files (default: github_pr_dataset)'
    )
    parser.add_argument(
        '--pattern',
        default='*_pr_*.json',
        help='File pattern to match (default: *_pr_*.json)'
    )
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Rebuild aggregate files after cleaning'
    )
    
    args = parser.parse_args()
    
    print("╔════════════════════════════════════════════════════════╗")
    print("║  PR JSON File Cleaner - Remove HTML Comments          ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    
    # Clean individual files
    processed, modified = clean_directory(args.directory, args.pattern)
    
    print()
    print("═" * 60)
    print(f"Summary:")
    print(f"  Files processed: {processed}")
    print(f"  Files modified:  {modified}")
    print(f"  Files unchanged: {processed - modified}")
    print("═" * 60)
    
    # Rebuild aggregate files if requested
    if args.rebuild:
        clean_aggregate_files(args.directory)
        print()
        print("✓ Aggregate files rebuilt successfully!")


if __name__ == '__main__':
    main()
