"""
BLEU Score Evaluation Script
Computes BLEU scores for generated titles and descriptions against reference texts.
"""

import os
import argparse
from typing import List, Dict
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import numpy as np
import json


def ensure_nltk_data():
    """Download necessary NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')


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
    """Tokenize text into words."""
    try:
        return word_tokenize(text.lower())
    except:
        # Fallback to simple split if tokenization fails
        return text.lower().split()


def compute_bleu_scores(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Compute BLEU scores for a list of reference and hypothesis pairs.
    
    Args:
        references: List of reference texts
        hypotheses: List of generated/hypothesis texts
    
    Returns:
        Dictionary containing BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"Mismatch: {len(references)} references but {len(hypotheses)} hypotheses")
    
    smoothing = SmoothingFunction()
    
    # Tokenize all texts
    tokenized_refs = [[tokenize_text(ref)] for ref in references]
    tokenized_hyps = [tokenize_text(hyp) for hyp in hypotheses]
    
    # Compute individual BLEU scores
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []
    
    for ref, hyp in zip(tokenized_refs, tokenized_hyps):
        # BLEU-1
        bleu_1 = sentence_bleu(ref, hyp, weights=(1, 0, 0, 0), 
                               smoothing_function=smoothing.method1)
        bleu_1_scores.append(bleu_1)
        
        # BLEU-2
        bleu_2 = sentence_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0),
                               smoothing_function=smoothing.method1)
        bleu_2_scores.append(bleu_2)
        
        # BLEU-3
        bleu_3 = sentence_bleu(ref, hyp, weights=(0.33, 0.33, 0.33, 0),
                               smoothing_function=smoothing.method1)
        bleu_3_scores.append(bleu_3)
        
        # BLEU-4
        bleu_4 = sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25),
                               smoothing_function=smoothing.method1)
        bleu_4_scores.append(bleu_4)
    
    # Compute corpus-level BLEU scores
    corpus_bleu_1 = corpus_bleu(tokenized_refs, tokenized_hyps, 
                                weights=(1, 0, 0, 0),
                                smoothing_function=smoothing.method1)
    corpus_bleu_2 = corpus_bleu(tokenized_refs, tokenized_hyps,
                                weights=(0.5, 0.5, 0, 0),
                                smoothing_function=smoothing.method1)
    corpus_bleu_3 = corpus_bleu(tokenized_refs, tokenized_hyps,
                                weights=(0.33, 0.33, 0.33, 0),
                                smoothing_function=smoothing.method1)
    corpus_bleu_4 = corpus_bleu(tokenized_refs, tokenized_hyps,
                                weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=smoothing.method1)
    
    return {
        'bleu_1_mean': np.mean(bleu_1_scores),
        'bleu_2_mean': np.mean(bleu_2_scores),
        'bleu_3_mean': np.mean(bleu_3_scores),
        'bleu_4_mean': np.mean(bleu_4_scores),
        'bleu_1_std': np.std(bleu_1_scores),
        'bleu_2_std': np.std(bleu_2_scores),
        'bleu_3_std': np.std(bleu_3_scores),
        'bleu_4_std': np.std(bleu_4_scores),
        'corpus_bleu_1': corpus_bleu_1,
        'corpus_bleu_2': corpus_bleu_2,
        'corpus_bleu_3': corpus_bleu_3,
        'corpus_bleu_4': corpus_bleu_4,
    }


def main():
    """Main function to compute BLEU scores for titles and descriptions."""
    # Ensure NLTK data is available
    ensure_nltk_data()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Compute BLEU scores for generated PR titles and descriptions')
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
    
    # Compute BLEU scores for titles
    print("=" * 70)
    print("BLEU SCORES FOR TITLES")
    print("=" * 70)
    try:
        title_scores = compute_bleu_scores(reference_titles, generated_titles)
        print(f"Number of title pairs evaluated: {len(generated_titles)}")
        print(f"\nSentence-level BLEU scores (mean ± std):")
        print(f"  BLEU-1: {title_scores['bleu_1_mean']:.4f} ± {title_scores['bleu_1_std']:.4f}")
        print(f"  BLEU-2: {title_scores['bleu_2_mean']:.4f} ± {title_scores['bleu_2_std']:.4f}")
        print(f"  BLEU-3: {title_scores['bleu_3_mean']:.4f} ± {title_scores['bleu_3_std']:.4f}")
        print(f"  BLEU-4: {title_scores['bleu_4_mean']:.4f} ± {title_scores['bleu_4_std']:.4f}")
        print(f"\nCorpus-level BLEU scores:")
        print(f"  Corpus BLEU-1: {title_scores['corpus_bleu_1']:.4f}")
        print(f"  Corpus BLEU-2: {title_scores['corpus_bleu_2']:.4f}")
        print(f"  Corpus BLEU-3: {title_scores['corpus_bleu_3']:.4f}")
        print(f"  Corpus BLEU-4: {title_scores['corpus_bleu_4']:.4f}")
    except Exception as e:
        print(f"Error computing BLEU scores for titles: {e}")
    
    print()
    
    # Compute BLEU scores for descriptions
    print("=" * 70)
    print("BLEU SCORES FOR DESCRIPTIONS")
    print("=" * 70)
    try:
        description_scores = compute_bleu_scores(reference_descriptions, generated_descriptions)
        print(f"Number of description pairs evaluated: {len(generated_descriptions)}")
        print(f"\nSentence-level BLEU scores (mean ± std):")
        print(f"  BLEU-1: {description_scores['bleu_1_mean']:.4f} ± {description_scores['bleu_1_std']:.4f}")
        print(f"  BLEU-2: {description_scores['bleu_2_mean']:.4f} ± {description_scores['bleu_2_std']:.4f}")
        print(f"  BLEU-3: {description_scores['bleu_3_mean']:.4f} ± {description_scores['bleu_3_std']:.4f}")
        print(f"  BLEU-4: {description_scores['bleu_4_mean']:.4f} ± {description_scores['bleu_4_std']:.4f}")
        print(f"\nCorpus-level BLEU scores:")
        print(f"  Corpus BLEU-1: {description_scores['corpus_bleu_1']:.4f}")
        print(f"  Corpus BLEU-2: {description_scores['corpus_bleu_2']:.4f}")
        print(f"  Corpus BLEU-3: {description_scores['corpus_bleu_3']:.4f}")
        print(f"  Corpus BLEU-4: {description_scores['corpus_bleu_4']:.4f}")
    except Exception as e:
        print(f"Error computing BLEU scores for descriptions: {e}")
    
    print()
    
    # Save results to JSON
    results = {
        'titles': title_scores if 'title_scores' in locals() else None,
        'descriptions': description_scores if 'description_scores' in locals() else None,
    }
    
    output_path = os.path.join(output_dir, 'bleu_scores.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
