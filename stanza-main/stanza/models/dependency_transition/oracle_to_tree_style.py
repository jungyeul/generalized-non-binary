"""
Oracle to Tree-Style Dependency Conversion

This module implements the conversion from arc-standard oracle transitions
to tree-style dependency format (MRG-like format for training).

Pipeline:
    UD_English-Penn (non-projective)
    → pseudo-projective lifting φ(D) [Ai Ka's projectivization]
    → UD_to_oracle() [Ai Ka's function - extract transitions]
    → oracle_to_tree_style() [This file - build tree]
    → Tree-style dependency (MRG format)

Paper Format (paper_stepbystep.md):
- N-ary tree structure (not binary)
- Head word as STRING terminal (not wrapped in Tree)
- Format: (POS left_deps... word right_deps...)

Key Design (from Section 3.4):
- SHIFT: Token becomes a subtree (POS [word]) where word is STRING terminal
- LEFTARC: Add dependent as sibling BEFORE head word
- RIGHTARC: Add dependent as sibling AFTER head word

Pseudo-projective lifting (Section 3.4):
- Non-projective arc (h,d,ℓ) → (h',d,ℓ') where h' is ancestor of h
- Enriched label ℓ' encodes lifting path
- Deterministic decoding: φ⁻¹(φ(D)) = D (lossless)

Reference: TACL Paper Section 3.4 + Appendix (paper_stepbystep.md)
"""

from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from nltk.tree import Tree
import logging

logger = logging.getLogger('stanza')


def oracle_to_tree_style(tokens: List[Dict], actions: List[str], 
                         pos_key: str = "xpos",
                         include_deprel: bool = False) -> Tree:
    """
    Build a tree-style dependency structure from oracle actions.
    
    Paper Format (paper_stepbystep.md):
    - SHIFT: Token → (POS [word]) where word is STRING terminal
    - LEFTARC: Add dependent as sibling BEFORE head word
    - RIGHTARC: Add dependent as sibling AFTER head word
    
    Result: N-ary tree with head word as terminal string
    Format: (POS left_deps... word right_deps...)
    
    Example: "book me the morning flight"
        → (ROOT (VB book (PRP me) (NN (DT the) (NN morning) flight)))
    
    Args:
        tokens: List of token dicts with 'id', 'form', 'xpos'/'upos', optionally 'deprel'
        actions: List of parser actions: 'SHIFT', 'LEFTARC', 'RIGHTARC'
        pos_key: Key for POS tag ('xpos' or 'upos')
        include_deprel: Whether to include dependency relation labels (not yet implemented)
    
    Returns:
        nltk.Tree: Tree-style dependency structure in paper format
    """
    if not tokens:
        return Tree("ROOT", [])
    
    # Build token lookup (id -> token dict)
    token_map = {}
    for tok in tokens:
        tid = int(tok['id']) if isinstance(tok['id'], str) else tok['id']
        token_map[tid] = tok
    
    # Stack holds tuples: (label, children_list)
    # children_list contains: strings (head word) and Trees (dependents)
    # Use None as placeholder for ROOT position
    stack = [None]
    
    # Buffer holds token IDs to be shifted
    buffer = [int(tok['id']) if isinstance(tok['id'], str) else tok['id'] 
              for tok in tokens]
    
    def to_tree(label, children):
        """Convert (label, children) tuple to Tree."""
        return Tree(label, children)
    
    for action in actions:
        if action == "SHIFT":
            # SHIFT: Token becomes subtree (POS [word])
            # Word is a string in the children list
            if not buffer:
                raise ValueError("Cannot SHIFT with empty buffer")
            
            token_id = buffer.pop(0)
            tok = token_map[token_id]
            
            # Get POS tag
            pos = tok.get(pos_key, tok.get('upos', tok.get('xpos', 'X')))
            if pos is None:
                pos = 'X'
            form = tok.get('form', str(token_id))
            
            # Create subtree: (POS [word]) - word is STRING terminal
            stack.append((pos, [form]))
            
        elif action == "LEFTARC":
            # LEFTARC: Add dependent as sibling BEFORE head word
            # stack: [..., T₁, T₂] → [..., T₂']
            if len(stack) < 2:
                raise ValueError("Cannot LEFTARC with stack < 2")
            
            head_item = stack[-1]      # head (stays)
            dep_item = stack[-2]       # dependent (removed)
            
            # Check if dep_item is ROOT placeholder (None)
            if dep_item is None:
                raise ValueError("Cannot LEFTARC with ROOT as dependent")
            
            head_label, head_children = head_item
            dep_label, dep_children = dep_item
            
            # Convert dependent to Tree
            dep_tree = to_tree(dep_label, dep_children)
            
            # Paper format: add dep to LEFT of head's children
            new_children = [dep_tree] + head_children
            
            stack.pop(-2)
            stack[-1] = (head_label, new_children)
            
        elif action == "RIGHTARC":
            # RIGHTARC: Add dependent as sibling AFTER head word
            # stack: [..., T₁, T₂] → [..., T₁']
            if len(stack) < 2:
                raise ValueError("Cannot RIGHTARC with stack < 2")
            
            head_item = stack[-2]      # head (stays)
            dep_item = stack[-1]       # dependent (removed)
            
            # Check if head_item is ROOT placeholder (None)
            if head_item is None:
                # This is attachment to ROOT
                dep_label, dep_children = dep_item
                dep_tree = to_tree(dep_label, dep_children)
                stack.pop()
                stack[-1] = ("ROOT", [dep_tree])
            else:
                head_label, head_children = head_item
                dep_label, dep_children = dep_item
                
                # Convert dependent to Tree
                dep_tree = to_tree(dep_label, dep_children)
                
                # Paper format: add dep to RIGHT of head's children
                new_children = head_children + [dep_tree]
                
                stack.pop()
                stack[-1] = (head_label, new_children)
            
        else:
            raise ValueError(f"Unknown action: {action}")
    
    # Build final tree
    if len(stack) == 2 and stack[0] is None:
        label, children = stack[1]
        return Tree("ROOT", [to_tree(label, children)])
    elif len(stack) == 1 and isinstance(stack[0], tuple):
        label, children = stack[0]
        return to_tree(label, children)
    else:
        # Try to return whatever we have
        final_children = []
        for item in stack:
            if item is None:
                continue
            if isinstance(item, tuple):
                label, children = item
                final_children.append(to_tree(label, children))
            elif isinstance(item, Tree):
                final_children.append(item)
        return Tree("ROOT", final_children)


def _get_head_word(tree: Tree) -> Optional[str]:
    """Get the head word (the STRING terminal) from a subtree."""
    if not isinstance(tree, Tree):
        return None
    for child in tree:
        if isinstance(child, str):
            return child
    # No direct string child, search recursively
    for child in tree:
        if isinstance(child, Tree):
            result = _get_head_word(child)
            if result:
                return result
    return None


def tree_style_to_conllu(tree: Tree, pos_key: str = "xpos") -> List[Dict]:
    """
    Convert tree-style dependency back to CONLLU format.
    
    Paper Format: (POS left_deps... word right_deps...)
    - Head word is the STRING terminal
    - Dependencies are Tree subtrees
    
    Args:
        tree: Tree-style dependency tree in paper format
        pos_key: Key for POS tag in output
    
    Returns:
        List of token dicts with head information
    """
    tokens = []
    
    def count_words_in_subtree(node) -> int:
        """Count words in a subtree."""
        if isinstance(node, str):
            return 1
        if not isinstance(node, Tree):
            return 0
        count = 0
        for child in node:
            count += count_words_in_subtree(child)
        return count
    
    def process_node(node: Tree, head_word_idx: int = -1, word_offset: int = 0):
        """Process a tree node and extract tokens."""
        if not isinstance(node, Tree):
            return
        
        label = node.label()
        
        if label == "ROOT":
            offset = word_offset
            for child in node:
                if isinstance(child, Tree):
                    process_node(child, -1, offset)
                    offset += count_words_in_subtree(child)
            return
        
        # Find head word index (the STRING child)
        head_child_idx = -1
        for i, child in enumerate(node):
            if isinstance(child, str):
                head_child_idx = i
                break
        
        if head_child_idx == -1:
            # No string child - fallback
            offset = word_offset
            for child in node:
                if isinstance(child, Tree):
                    process_node(child, head_word_idx, offset)
                    offset += count_words_in_subtree(child)
            return
        
        # Calculate this node's head word absolute index
        words_before_head = 0
        for i, child in enumerate(node):
            if i < head_child_idx:
                words_before_head += count_words_in_subtree(child)
        
        my_head_abs_idx = word_offset + words_before_head
        
        # Process children
        curr_offset = word_offset
        for i, child in enumerate(node):
            if isinstance(child, str):
                # This is the head word
                tokens.append({
                    'id': str(len(tokens) + 1),
                    'form': child,
                    pos_key: label,
                    'head': str(head_word_idx + 1) if head_word_idx >= 0 else '0',
                    'deprel': 'root' if head_word_idx < 0 else 'dep'
                })
                curr_offset += 1
            elif isinstance(child, Tree):
                # This is a dependent subtree
                process_node(child, my_head_abs_idx, curr_offset)
                curr_offset += count_words_in_subtree(child)
    
    # Process the tree
    process_node(tree, -1, 0)
    
    # Sort by ID
    tokens.sort(key=lambda t: int(t['id']))
    return tokens


def conllu_to_tree_style_pipeline(tokens: List[Dict], 
                                   pos_key: str = "xpos",
                                   include_deprel: bool = False) -> Tuple[Tree, List[str]]:
    """
    Complete pipeline: CONLLU → Oracle → Tree-style
    
    Uses Ai Ka's UD_to_oracle function and then converts to tree-style.
    
    Args:
        tokens: List of token dicts from CONLLU
        pos_key: Key for POS tag
        include_deprel: Whether to include dependency relation labels
    
    Returns:
        Tuple of (Tree, oracle_actions)
    """
    # Import Ai Ka's function
    from transition_based_parser import UD_to_oracle
    
    # Convert tokens to sentence dict format expected by UD_to_oracle
    sentence = {int(t['id']): t for t in tokens}
    
    # Get oracle steps
    oracle_steps = UD_to_oracle(sentence)
    
    # Extract just the actions
    actions = [step[2] for step in oracle_steps]
    
    # Convert to tree-style
    tree = oracle_to_tree_style(tokens, actions, pos_key, include_deprel)
    
    return tree, actions


def verify_roundtrip(tokens: List[Dict], pos_key: str = "xpos") -> bool:
    """
    Verify that CONLLU → Oracle → Tree-style → CONLLU produces same heads.
    
    Args:
        tokens: Original CONLLU tokens
        pos_key: POS tag key
    
    Returns:
        True if roundtrip is successful
    """
    try:
        tree, actions = conllu_to_tree_style_pipeline(tokens, pos_key, include_deprel=True)
        reconstructed = tree_style_to_conllu(tree, pos_key)
        
        # Compare heads
        orig_heads = {int(t['id']): int(t['head']) for t in tokens}
        recon_heads = {int(t['id']): int(t['head']) for t in reconstructed}
        
        if orig_heads == recon_heads:
            return True
        else:
            logger.warning(f"Head mismatch: {orig_heads} vs {recon_heads}")
            return False
    except Exception as e:
        logger.error(f"Roundtrip failed: {e}")
        return False


def demo():
    """Demonstrate the oracle-to-tree-style conversion."""
    
    # Example: "The cat sat"
    # 1:The -> 2:cat (det)
    # 2:cat -> 3:sat (nsubj)
    # 3:sat -> 0:ROOT (root)
    
    tokens = [
        {'id': '1', 'form': 'The', 'xpos': 'DT', 'upos': 'DET', 'head': '2', 'deprel': 'det'},
        {'id': '2', 'form': 'cat', 'xpos': 'NN', 'upos': 'NOUN', 'head': '3', 'deprel': 'nsubj'},
        {'id': '3', 'form': 'sat', 'xpos': 'VBD', 'upos': 'VERB', 'head': '0', 'deprel': 'root'},
    ]
    
    # Oracle from Ai Ka's function would give:
    # SHIFT (buffer=[1,2,3], stack=[0]) -> stack=[0,1]
    # SHIFT (buffer=[2,3], stack=[0,1]) -> stack=[0,1,2]
    # LEFTARC (1 is dep of 2) -> stack=[0,2] (1 attached to 2)
    # SHIFT (buffer=[3], stack=[0,2]) -> stack=[0,2,3]
    # LEFTARC (2 is dep of 3) -> stack=[0,3] (2 attached to 3)
    # RIGHTARC (3 is dep of 0/ROOT) -> stack=[0] (3 attached to ROOT)
    
    actions = ['SHIFT', 'SHIFT', 'LEFTARC', 'SHIFT', 'LEFTARC', 'RIGHTARC']
    
    print("=" * 60)
    print("Input tokens:")
    for t in tokens:
        print(f"  {t['id']}: {t['form']} ({t['xpos']}) -> {t['head']} ({t['deprel']})")
    
    print("\n" + "=" * 60)
    print("Oracle actions:")
    print(f"  {actions}")
    
    print("\n" + "=" * 60)
    print("Tree-style (UAS only):")
    tree_uas = oracle_to_tree_style(tokens, actions, pos_key='xpos', include_deprel=False)
    print(f"  {tree_uas}")
    tree_uas.pretty_print()
    
    print("\n" + "=" * 60)
    print("Tree-style (with deprel):")
    tree_full = oracle_to_tree_style(tokens, actions, pos_key='xpos', include_deprel=True)
    print(f"  {tree_full}")
    tree_full.pretty_print()
    
    print("\n" + "=" * 60)
    print("One-line format (for training data):")
    tree_str = str(tree_uas).replace('\n', ' ')
    tree_str = ' '.join(tree_str.split())
    print(f"  {tree_str}")
    
    return tree_full


def demo_with_aika_function():
    """Demo using Ai Ka's UD_to_oracle function directly."""
    import sys
    sys.path.insert(0, '/home/zihopc/vscode/20251116_stanza')
    
    from transition_based_parser import UD_to_oracle
    
    # Same example
    sentence = {
        1: {'id': 1, 'form': 'The', 'xpos': 'DT', 'upos': 'DET', 'head': 2, 'deprel': 'det'},
        2: {'id': 2, 'form': 'cat', 'xpos': 'NN', 'upos': 'NOUN', 'head': 3, 'deprel': 'nsubj'},
        3: {'id': 3, 'form': 'sat', 'xpos': 'VBD', 'upos': 'VERB', 'head': 0, 'deprel': 'root'},
    }
    
    print("=" * 60)
    print("Using Ai Ka's UD_to_oracle function:")
    
    oracle_steps = UD_to_oracle(sentence)
    
    print("\nOracle steps:")
    for step in oracle_steps:
        buffer, stack, action = step
        print(f"  Buffer: {buffer}, Stack: {stack}, Action: {action}")
    
    actions = [step[2] for step in oracle_steps]
    
    # Convert to list of dicts for our function
    tokens = [{'id': str(k), **{kk: vv for kk, vv in v.items() if kk != 'id'}} 
              for k, v in sentence.items()]
    
    print("\n" + "=" * 60)
    print("Tree-style result:")
    tree = oracle_to_tree_style(tokens, actions, pos_key='xpos', include_deprel=False)
    tree.pretty_print()
    
    return tree


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Demo 1: Manual oracle")
    print("=" * 70)
    demo()
    
    print("\n\n")
    print("Demo 2: Using Ai Ka's function")
    print("=" * 70)
    try:
        demo_with_aika_function()
    except ImportError as e:
        print(f"Could not import Ai Ka's function: {e}")
        print("Run from the correct directory or add to PYTHONPATH")
