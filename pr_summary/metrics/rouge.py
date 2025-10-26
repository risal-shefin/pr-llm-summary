"""
ROUGE Score Evaluation Script
Computes ROUGE scores for generated titles and descriptions against reference texts.
"""

import os
import argparse
from typing import List, Dict
import re
from collections import Counter
import json


def load_text_file(filepath: str) -> List[str]:
    """Load text file and return list of lines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines


def load_json_file(filepath: str) -> List[Dict]:
    """Load JSON file and return list of dictionaries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def tokenize_text(text: str) -> List[str]:
    """Simple tokenization: lowercase and split on whitespace/punctuation."""
    text = text.lower()
    # Split on whitespace and punctuation but keep words
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Generate n-grams from tokens."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return Counter(ngrams)


def rouge_n(reference: str, hypothesis: str, n: int) -> Dict[str, float]:
    """
    Compute ROUGE-N score.
    
    Args:
        reference: Reference text
        hypothesis: Generated text
        n: N-gram size (1 for unigrams, 2 for bigrams, etc.)
    
    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    ref_tokens = tokenize_text(reference)
    hyp_tokens = tokenize_text(hypothesis)
    
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    ref_ngrams = get_ngrams(ref_tokens, n)
    hyp_ngrams = get_ngrams(hyp_tokens, n)
    
    if len(ref_ngrams) == 0 or len(hyp_ngrams) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Count overlapping n-grams
    overlap = sum((ref_ngrams & hyp_ngrams).values())
    
    # Precision: overlap / total hypothesis n-grams
    precision = overlap / sum(hyp_ngrams.values()) if sum(hyp_ngrams.values()) > 0 else 0.0
    
    # Recall: overlap / total reference n-grams
    recall = overlap / sum(ref_ngrams.values()) if sum(ref_ngrams.values()) > 0 else 0.0
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def rouge_l(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute ROUGE-L score using Longest Common Subsequence (LCS).
    
    Args:
        reference: Reference text
        hypothesis: Generated text
    
    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    ref_tokens = tokenize_text(reference)
    hyp_tokens = tokenize_text(hypothesis)
    
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Compute LCS length using dynamic programming
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    # Precision: LCS / length of hypothesis
    precision = lcs_length / len(hyp_tokens) if len(hyp_tokens) > 0 else 0.0
    
    # Recall: LCS / length of reference
    recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_rouge_scores(references: List[str], hypotheses: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute ROUGE scores for a list of reference and hypothesis pairs.
    
    Args:
        references: List of reference texts
        hypotheses: List of generated/hypothesis texts
    
    Returns:
        Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"Mismatch: {len(references)} references but {len(hypotheses)} hypotheses")
    
    rouge_1_scores = {'precision': [], 'recall': [], 'f1': []}
    rouge_2_scores = {'precision': [], 'recall': [], 'f1': []}
    rouge_l_scores = {'precision': [], 'recall': [], 'f1': []}
    
    for ref, hyp in zip(references, hypotheses):
        # ROUGE-1
        r1 = rouge_n(ref, hyp, 1)
        rouge_1_scores['precision'].append(r1['precision'])
        rouge_1_scores['recall'].append(r1['recall'])
        rouge_1_scores['f1'].append(r1['f1'])
        
        # ROUGE-2
        r2 = rouge_n(ref, hyp, 2)
        rouge_2_scores['precision'].append(r2['precision'])
        rouge_2_scores['recall'].append(r2['recall'])
        rouge_2_scores['f1'].append(r2['f1'])
        
        # ROUGE-L
        rl = rouge_l(ref, hyp)
        rouge_l_scores['precision'].append(rl['precision'])
        rouge_l_scores['recall'].append(rl['recall'])
        rouge_l_scores['f1'].append(rl['f1'])
    
    # Compute averages
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0
    
    def std(lst):
        if not lst:
            return 0.0
        mean = avg(lst)
        variance = sum((x - mean) ** 2 for x in lst) / len(lst)
        return variance ** 0.5
    
    return {
        'rouge_1': {
            'precision_mean': avg(rouge_1_scores['precision']),
            'precision_std': std(rouge_1_scores['precision']),
            'recall_mean': avg(rouge_1_scores['recall']),
            'recall_std': std(rouge_1_scores['recall']),
            'f1_mean': avg(rouge_1_scores['f1']),
            'f1_std': std(rouge_1_scores['f1']),
        },
        'rouge_2': {
            'precision_mean': avg(rouge_2_scores['precision']),
            'precision_std': std(rouge_2_scores['precision']),
            'recall_mean': avg(rouge_2_scores['recall']),
            'recall_std': std(rouge_2_scores['recall']),
            'f1_mean': avg(rouge_2_scores['f1']),
            'f1_std': std(rouge_2_scores['f1']),
        },
        'rouge_l': {
            'precision_mean': avg(rouge_l_scores['precision']),
            'precision_std': std(rouge_l_scores['precision']),
            'recall_mean': avg(rouge_l_scores['recall']),
            'recall_std': std(rouge_l_scores['recall']),
            'f1_mean': avg(rouge_l_scores['f1']),
            'f1_std': std(rouge_l_scores['f1']),
        }
    }


def main():
    """Main function to compute ROUGE scores for titles and descriptions."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Compute ROUGE scores for generated PR titles and descriptions')
    parser.add_argument('--output-dir', type=str, default='trial_output',
                        help='Directory containing generated_outputs.json and reference_outputs.json (default: trial_output)')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    
    generated_outputs_path = os.path.join(output_dir, 'generated_outputs.json')
    reference_outputs_path = os.path.join(output_dir, 'reference_outputs.json')
    
    # Load JSON data
    print("Loading data...")
    generated_data = load_json_file(generated_outputs_path)
    reference_data = load_json_file(reference_outputs_path)
    
    print(f"Loaded {len(generated_data)} generated outputs")
    print(f"Loaded {len(reference_data)} reference outputs")
    
    # Extract titles and descriptions
    generated_titles = [item.get('generated_title', '') for item in generated_data]
    reference_titles = [item.get('reference_title', '') for item in reference_data]
    generated_descriptions = [item.get('generated_description', '') for item in generated_data]
    reference_descriptions = [item.get('reference_description', '') for item in reference_data]
    
    print(f"Extracted {len(generated_titles)} generated titles")
    print(f"Extracted {len(reference_titles)} reference titles")
    print(f"Extracted {len(generated_descriptions)} generated descriptions")
    print(f"Extracted {len(reference_descriptions)} reference descriptions")
    print()
    
    # Compute ROUGE scores for titles
    print("=" * 70)
    print("ROUGE SCORES FOR TITLES")
    print("=" * 70)
    try:
        title_scores = compute_rouge_scores(reference_titles, generated_titles)
        print(f"Number of title pairs evaluated: {len(generated_titles)}")
        
        print(f"\nROUGE-1 (Unigram overlap):")
        print(f"  Precision: {title_scores['rouge_1']['precision_mean']:.4f} ± {title_scores['rouge_1']['precision_std']:.4f}")
        print(f"  Recall:    {title_scores['rouge_1']['recall_mean']:.4f} ± {title_scores['rouge_1']['recall_std']:.4f}")
        print(f"  F1:        {title_scores['rouge_1']['f1_mean']:.4f} ± {title_scores['rouge_1']['f1_std']:.4f}")
        
        print(f"\nROUGE-2 (Bigram overlap):")
        print(f"  Precision: {title_scores['rouge_2']['precision_mean']:.4f} ± {title_scores['rouge_2']['precision_std']:.4f}")
        print(f"  Recall:    {title_scores['rouge_2']['recall_mean']:.4f} ± {title_scores['rouge_2']['recall_std']:.4f}")
        print(f"  F1:        {title_scores['rouge_2']['f1_mean']:.4f} ± {title_scores['rouge_2']['f1_std']:.4f}")
        
        print(f"\nROUGE-L (Longest common subsequence):")
        print(f"  Precision: {title_scores['rouge_l']['precision_mean']:.4f} ± {title_scores['rouge_l']['precision_std']:.4f}")
        print(f"  Recall:    {title_scores['rouge_l']['recall_mean']:.4f} ± {title_scores['rouge_l']['recall_std']:.4f}")
        print(f"  F1:        {title_scores['rouge_l']['f1_mean']:.4f} ± {title_scores['rouge_l']['f1_std']:.4f}")
    except Exception as e:
        print(f"Error computing ROUGE scores for titles: {e}")
    
    print()
    
    # Compute ROUGE scores for descriptions
    print("=" * 70)
    print("ROUGE SCORES FOR DESCRIPTIONS")
    print("=" * 70)
    try:
        description_scores = compute_rouge_scores(reference_descriptions, generated_descriptions)
        print(f"Number of description pairs evaluated: {len(generated_descriptions)}")
        
        print(f"\nROUGE-1 (Unigram overlap):")
        print(f"  Precision: {description_scores['rouge_1']['precision_mean']:.4f} ± {description_scores['rouge_1']['precision_std']:.4f}")
        print(f"  Recall:    {description_scores['rouge_1']['recall_mean']:.4f} ± {description_scores['rouge_1']['recall_std']:.4f}")
        print(f"  F1:        {description_scores['rouge_1']['f1_mean']:.4f} ± {description_scores['rouge_1']['f1_std']:.4f}")
        
        print(f"\nROUGE-2 (Bigram overlap):")
        print(f"  Precision: {description_scores['rouge_2']['precision_mean']:.4f} ± {description_scores['rouge_2']['precision_std']:.4f}")
        print(f"  Recall:    {description_scores['rouge_2']['recall_mean']:.4f} ± {description_scores['rouge_2']['recall_std']:.4f}")
        print(f"  F1:        {description_scores['rouge_2']['f1_mean']:.4f} ± {description_scores['rouge_2']['f1_std']:.4f}")
        
        print(f"\nROUGE-L (Longest common subsequence):")
        print(f"  Precision: {description_scores['rouge_l']['precision_mean']:.4f} ± {description_scores['rouge_l']['precision_std']:.4f}")
        print(f"  Recall:    {description_scores['rouge_l']['recall_mean']:.4f} ± {description_scores['rouge_l']['recall_std']:.4f}")
        print(f"  F1:        {description_scores['rouge_l']['f1_mean']:.4f} ± {description_scores['rouge_l']['f1_std']:.4f}")
    except Exception as e:
        print(f"Error computing ROUGE scores for descriptions: {e}")
    
    print()
    
    # Save results to JSON
    results = {
        'titles': title_scores if 'title_scores' in locals() else None,
        'descriptions': description_scores if 'description_scores' in locals() else None,
    }
    
    output_path = os.path.join(output_dir, 'rouge_scores.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
