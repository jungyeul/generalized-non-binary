"""
CONLLU to Tree-Style Conversion Pipeline

Complete pipeline for converting CONLLU files to tree-style dependency format:

    UD_English-Penn_pseudo (CONLLU) 
    → UD_to_oracle() [Ai Ka's function - extract transitions]
    → oracle_to_tree_style() [build tree structure]
    → Tree-style dependency (MRG format for training)

Usage:
    python conllu_to_mrg_pipeline.py input.conllu output.mrg

Author: Your implementation based on advisor's TACL paper
"""

import sys
import os
import argparse
import logging
from typing import List, Dict, Tuple, Optional

# Add parent paths for imports
sys.path.insert(0, '/home/zihopc/vscode/20251116_stanza')
sys.path.insert(0, '/home/zihopc/vscode/20251116_stanza/stanza-main')

from nltk.tree import Tree
from transition_based_parser import UD_to_oracle, oracle_to_UD
from stanza.models.dependency_transition.oracle_to_tree_style import oracle_to_tree_style

logger = logging.getLogger('stanza')


def read_conllu(file_path: str) -> List[Dict[int, Dict]]:
    """
    Read CONLLU file and return list of sentences.
    
    Each sentence is a dict: {token_id: token_dict}
    """
    sentences = []
    current_sentence = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                # End of sentence
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = {}
                continue
            
            if line.startswith('#'):
                # Comment line - skip
                continue
            
            parts = line.split('\t')
            if len(parts) < 10:
                continue
            
            # Skip multi-word tokens (e.g., "1-2") and empty nodes (e.g., "1.1")
            if '-' in parts[0] or '.' in parts[0]:
                continue
            
            token_id = int(parts[0])
            token = {
                'id': parts[0],  # Keep as string for Ai Ka's function
                'form': parts[1],
                'lemma': parts[2],
                'upos': parts[3],
                'xpos': parts[4],
                'feats': parts[5],
                'head': parts[6],
                'deprel': parts[7],
                'deps': parts[8],
                'misc': parts[9]
            }
            current_sentence[token_id] = token
        
        # Don't forget last sentence
        if current_sentence:
            sentences.append(current_sentence)
    
    return sentences


def sentence_to_tree_style(sentence: Dict[int, Dict], 
                           pos_key: str = "xpos",
                           include_deprel: bool = False) -> Optional[Tree]:
    """
    Convert a single CONLLU sentence to tree-style format.
    
    Args:
        sentence: Dict mapping token_id to token_dict
        pos_key: 'xpos' or 'upos'
        include_deprel: Whether to include dependency relation labels
    
    Returns:
        nltk.Tree or None if conversion fails
    """
    try:
        # Get oracle steps using Ai Ka's function
        oracle_steps = UD_to_oracle(sentence)
        
        # Extract actions
        actions = [step[2] for step in oracle_steps]
        
        # Convert to token list for our function
        tokens = list(sentence.values())
        
        # Build tree-style
        tree = oracle_to_tree_style(tokens, actions, pos_key, include_deprel)
        
        return tree
    
    except Exception as e:
        logger.warning(f"Failed to convert sentence: {e}")
        return None


def convert_conllu_to_mrg(input_path: str, 
                          output_path: str,
                          pos_key: str = "xpos",
                          include_deprel: bool = False,
                          skip_errors: bool = True) -> Tuple[int, int]:
    """
    Convert CONLLU file to MRG format (tree-style dependency).
    
    Args:
        input_path: Path to input CONLLU file
        output_path: Path to output MRG file
        pos_key: 'xpos' or 'upos' for POS tags
        include_deprel: Whether to include dependency relations
        skip_errors: If True, skip sentences that fail to convert
    
    Returns:
        Tuple of (successful_count, failed_count)
    """
    sentences = read_conllu(input_path)
    
    successful = 0
    failed = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(sentences):
            tree = sentence_to_tree_style(sentence, pos_key, include_deprel)
            
            if tree is not None:
                # Write as one line (MRG format)
                tree_str = str(tree).replace('\n', ' ')
                tree_str = ' '.join(tree_str.split())
                f.write(tree_str + '\n')
                successful += 1
            else:
                failed += 1
                if not skip_errors:
                    raise ValueError(f"Failed to convert sentence {i+1}")
    
    logger.info(f"Converted {successful} sentences, {failed} failed")
    return successful, failed


def verify_conversion(input_path: str, 
                      pos_key: str = "xpos",
                      sample_size: int = 10) -> None:
    """
    Verify conversion by checking a sample of sentences.
    
    Shows the original CONLLU, oracle steps, and resulting tree.
    """
    sentences = read_conllu(input_path)
    
    for i, sentence in enumerate(sentences[:sample_size]):
        print(f"\n{'='*60}")
        print(f"Sentence {i+1}:")
        
        # Print original
        print("\nOriginal CONLLU:")
        for tid, tok in sorted(sentence.items()):
            print(f"  {tok['id']}: {tok['form']} ({tok[pos_key]}) -> {tok['head']} ({tok['deprel']})")
        
        # Get oracle
        try:
            oracle_steps = UD_to_oracle(sentence)
            print("\nOracle steps:")
            for step in oracle_steps:
                buffer, stack, action = step
                print(f"  Buffer: {buffer}, Stack: {stack}, Action: {action}")
            
            # Convert to tree
            actions = [step[2] for step in oracle_steps]
            tokens = list(sentence.values())
            tree = oracle_to_tree_style(tokens, actions, pos_key, include_deprel=False)
            
            print("\nTree-style:")
            tree.pretty_print()
            
            print("\nMRG format:")
            tree_str = str(tree).replace('\n', ' ')
            print(' '.join(tree_str.split()))
            
        except Exception as e:
            print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CONLLU to tree-style MRG format"
    )
    parser.add_argument("input", help="Input CONLLU file")
    parser.add_argument("output", help="Output MRG file")
    parser.add_argument("--pos", choices=["xpos", "upos"], default="xpos",
                       help="POS tag type to use")
    parser.add_argument("--deprel", action="store_true",
                       help="Include dependency relation labels")
    parser.add_argument("--verify", action="store_true",
                       help="Verify conversion on sample sentences")
    parser.add_argument("--sample-size", type=int, default=10,
                       help="Number of sentences to verify")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.verify:
        verify_conversion(args.input, args.pos, args.sample_size)
    else:
        success, fail = convert_conllu_to_mrg(
            args.input, args.output, 
            args.pos, args.deprel
        )
        print(f"Done! Converted {success} sentences, {fail} failed")


if __name__ == "__main__":
    main()
