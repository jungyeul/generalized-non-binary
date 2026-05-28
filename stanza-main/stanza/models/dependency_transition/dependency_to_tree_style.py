"""
Dependency to Tree-Style Dependency Conversion

This module implements the conversion from standard dependency format (CONLLU)
to tree-style dependency format using left-arc and right-arc transitions.

The tree-style dependency format represents dependency trees as constituency-like
structures where each arc is represented as a subtree.

Reference: Overleaf Section 3.1 - Tree-style Dependency Definition

Author: Based on advisor's TACL paper on recasting dependency
"""

from collections import defaultdict
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from nltk.tree import Tree

logger = logging.getLogger('stanza')


@dataclass
class Token:
    """Represents a single token in dependency parsing."""
    id: int
    form: str
    upos: str
    xpos: str
    head: int
    deprel: str
    
    def __repr__(self):
        return f"Token({self.id}, '{self.form}', head={self.head})"


def read_conllu_sentence(lines: List[str]) -> List[Token]:
    """
    Parse CONLLU format lines into Token objects.
    
    Args:
        lines: List of CONLLU format lines (excluding comments and blank lines)
    
    Returns:
        List of Token objects
    """
    tokens = []
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.strip().split('\t')
        if len(parts) < 10:
            continue
        # Skip multi-word tokens (e.g., "1-2")
        if '-' in parts[0] or '.' in parts[0]:
            continue
        
        token = Token(
            id=int(parts[0]),
            form=parts[1],
            upos=parts[3],
            xpos=parts[4],
            head=int(parts[6]),
            deprel=parts[7]
        )
        tokens.append(token)
    return tokens


def dependency_to_tree_style(tokens: List[Token], pos_type: str = "XPOS", 
                              use_deprel: bool = False) -> Tree:
    """
    Convert dependency structure to tree-style representation.
    
    This function converts a standard dependency tree to a constituency-like
    tree structure where:
    - Each word becomes a preterminal (POS -> word)
    - Dependency arcs are represented as parent-child relationships in the tree
    - The head of each subtree is the word, and dependents are siblings
    
    Args:
        tokens: List of Token objects with head information
        pos_type: "XPOS" or "UPOS" for POS tags
        use_deprel: If True, use dependency relation as internal node label
                   If False, use a generic label (for UAS-only evaluation)
    
    Returns:
        nltk.Tree object representing the tree-style dependency
    
    Example:
        Input: "The cat sat"
               1:The -> 2:cat (det)
               2:cat -> 3:sat (nsubj)  
               3:sat -> 0:ROOT (root)
        
        Output (tree-style):
            (ROOT
              (root
                (nsubj
                  (det (DT The))
                  (NN cat))
                (VBD sat)))
    """
    if not tokens:
        return Tree("ROOT", [])
    
    # Build head-to-dependents mapping
    head_to_deps = defaultdict(list)
    for token in tokens:
        head_to_deps[token.head].append(token)
    
    # Sort dependents by their position
    for head_id in head_to_deps:
        head_to_deps[head_id].sort(key=lambda t: t.id)
    
    def build_subtree(token: Token) -> Tree:
        """Recursively build tree from a token and its dependents."""
        # Get POS tag
        pos = token.xpos if pos_type == "XPOS" else token.upos
        
        # Create leaf: (POS word)
        leaf = Tree(pos, [token.form])
        
        # Get dependents
        deps = head_to_deps.get(token.id, [])
        
        if not deps:
            # No dependents, return just the preterminal
            return leaf
        
        # Build children: dependents + head (in order by position)
        children = []
        all_nodes = [(token, leaf)] + [(d, build_subtree(d)) for d in deps]
        all_nodes.sort(key=lambda x: x[0].id)
        
        for node_token, node_tree in all_nodes:
            if node_token.id == token.id:
                # This is the head - add directly
                children.append(node_tree)
            else:
                # This is a dependent - wrap with deprel if needed
                if use_deprel:
                    children.append(Tree(node_token.deprel, [node_tree]))
                else:
                    # For UAS only - use generic "DEP" label
                    children.append(Tree("DEP", [node_tree]))
        
        # The whole subtree is labeled with the token's deprel to its head
        if use_deprel:
            return Tree(token.deprel, children)
        else:
            return Tree("X", children)  # Generic label for UAS
    
    # Find root(s) - tokens with head=0
    roots = [t for t in tokens if t.head == 0]
    
    if len(roots) == 1:
        root_tree = build_subtree(roots[0])
        return Tree("ROOT", [root_tree])
    elif len(roots) > 1:
        # Multiple roots (rare but possible)
        root_children = [build_subtree(r) for r in roots]
        return Tree("ROOT", root_children)
    else:
        raise ValueError("No root found in dependency tree")


def tree_style_to_dependency(tree: Tree, pos_type: str = "XPOS") -> List[Dict]:
    """
    Convert tree-style representation back to dependency format.
    
    This is the inverse of dependency_to_tree_style.
    Used for evaluation and debugging.
    
    Args:
        tree: nltk.Tree in tree-style format
        pos_type: "XPOS" or "UPOS" for output
    
    Returns:
        List of dicts with token information including head
    """
    tokens = []
    counter = [1]  # Mutable counter for word IDs
    
    def process_subtree(subtree: Tree, head_id: int = 0) -> Optional[int]:
        """
        Process a subtree and return the ID of its head word.
        """
        if not isinstance(subtree, Tree):
            return None
        
        # Find the preterminal (POS -> word) in this subtree
        # The head is the preterminal at the "root" level of this subtree
        head_word_id = None
        
        for child in subtree:
            if isinstance(child, Tree):
                if child.height() == 2:
                    # This is a preterminal (POS -> word)
                    if len(child) == 1 and isinstance(child[0], str):
                        word_id = counter[0]
                        counter[0] += 1
                        
                        token = {
                            "id": word_id,
                            "form": child[0],
                            "upos": child.label() if pos_type == "UPOS" else "_",
                            "xpos": child.label() if pos_type == "XPOS" else "_",
                            "head": head_id,
                            "deprel": subtree.label() if subtree.label() not in ["ROOT", "X", "DEP"] else "dep"
                        }
                        tokens.append(token)
                        
                        if head_word_id is None:
                            head_word_id = word_id
                else:
                    # Recursive subtree - process with current head as parent
                    child_head_id = process_subtree(child, head_word_id if head_word_id else head_id)
        
        return head_word_id
    
    # Handle ROOT wrapper
    if tree.label() == "ROOT" and len(tree) == 1:
        process_subtree(tree[0], 0)
    else:
        process_subtree(tree, 0)
    
    # Sort by ID
    tokens.sort(key=lambda t: t["id"])
    return tokens


def generate_arc_transitions(tokens: List[Token]) -> List[str]:
    """
    Generate the sequence of arc-standard transitions for a dependency tree.
    
    Transitions:
    - SHIFT: Move word from buffer to stack
    - LEFT-ARC: stack[-2] becomes dependent of stack[-1], pop stack[-2]
    - RIGHT-ARC: stack[-1] becomes dependent of stack[-2], pop stack[-1]
    
    This implements a static oracle for arc-standard parsing.
    
    Args:
        tokens: List of Token objects with head information
    
    Returns:
        List of transition strings
    """
    n = len(tokens)
    if n == 0:
        return []
    
    # Build dependency info
    heads = {t.id: t.head for t in tokens}
    
    # Track which tokens have all dependents attached
    dep_count = defaultdict(int)  # How many dependents each head needs
    for t in tokens:
        if t.head > 0:  # Not root
            dep_count[t.head] += 1
    
    attached = {t.id: False for t in tokens}  # Track if token has been attached to head
    
    transitions = []
    buffer = list(range(1, n + 1))  # Token IDs 1 to n
    stack = [0]  # Start with ROOT (0) on stack
    
    while buffer or len(stack) > 1:
        # Try LEFT-ARC: stack[-2] is dependent of stack[-1]
        if len(stack) >= 2:
            s1 = stack[-1]  # Top of stack
            s2 = stack[-2]  # Second on stack
            
            # Check if s2 should be attached to s1 (LEFT-ARC)
            if s2 > 0 and heads.get(s2) == s1:
                # Check if s2 has no more dependents to attach
                can_reduce = True
                for t in tokens:
                    if t.head == s2 and not attached.get(t.id, False):
                        can_reduce = False
                        break
                
                if can_reduce:
                    transitions.append("LEFT-ARC")
                    attached[s2] = True
                    stack.pop(-2)  # Remove s2
                    continue
            
            # Check if s1 should be attached to s2 (RIGHT-ARC)
            if s1 > 0 and heads.get(s1) == s2:
                # Check if s1 has no more dependents to attach
                can_reduce = True
                for t in tokens:
                    if t.head == s1 and not attached.get(t.id, False):
                        can_reduce = False
                        break
                
                if can_reduce:
                    transitions.append("RIGHT-ARC")
                    attached[s1] = True
                    stack.pop()  # Remove s1
                    continue
        
        # SHIFT if possible
        if buffer:
            transitions.append("SHIFT")
            stack.append(buffer.pop(0))
        else:
            # If we can't do anything, there might be an error
            logger.warning(f"Oracle stuck: stack={stack}, buffer={buffer}")
            break
    
    return transitions


def verify_transitions(tokens: List[Token], transitions: List[str]) -> bool:
    """
    Verify that executing the transitions produces the correct dependency tree.
    
    Args:
        tokens: Original tokens with head information
        transitions: List of transitions to verify
    
    Returns:
        True if transitions correctly reproduce the dependency tree
    """
    n = len(tokens)
    if n == 0:
        return len(transitions) == 0
    
    buffer = list(range(1, n + 1))
    stack = [0]
    arcs = []  # (head, dependent) pairs
    
    for trans in transitions:
        if trans == "SHIFT":
            if not buffer:
                return False
            stack.append(buffer.pop(0))
        elif trans == "LEFT-ARC":
            if len(stack) < 2:
                return False
            s1 = stack[-1]
            s2 = stack.pop(-2)
            arcs.append((s1, s2))
        elif trans == "RIGHT-ARC":
            if len(stack) < 2:
                return False
            s1 = stack.pop()
            s2 = stack[-1]
            arcs.append((s2, s1))
        else:
            logger.warning(f"Unknown transition: {trans}")
            return False
    
    # Check if arcs match original heads
    arc_dict = {dep: head for head, dep in arcs}
    
    for token in tokens:
        if token.id not in arc_dict:
            logger.warning(f"Token {token.id} not attached")
            return False
        if arc_dict[token.id] != token.head:
            logger.warning(f"Token {token.id}: expected head {token.head}, got {arc_dict[token.id]}")
            return False
    
    return True


# =============================================================================
# Convenience functions for working with CONLLU files
# =============================================================================

def conllu_to_tree_style_file(input_path: str, output_path: str, 
                               pos_type: str = "XPOS", 
                               use_deprel: bool = False):
    """
    Convert a CONLLU file to tree-style format (MRG-like).
    
    Args:
        input_path: Path to input CONLLU file
        output_path: Path to output file (one tree per line)
        pos_type: "XPOS" or "UPOS"
        use_deprel: Whether to include dependency relations
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sentences = content.strip().split('\n\n')
    trees = []
    
    for sent in sentences:
        lines = sent.strip().split('\n')
        tokens = read_conllu_sentence(lines)
        if tokens:
            tree = dependency_to_tree_style(tokens, pos_type, use_deprel)
            trees.append(tree)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for tree in trees:
            # Write as one line, similar to MRG format
            tree_str = str(tree).replace('\n', ' ')
            tree_str = ' '.join(tree_str.split())  # Normalize whitespace
            f.write(tree_str + '\n')
    
    logger.info(f"Converted {len(trees)} sentences to tree-style format")


def demo():
    """
    Demonstration of the dependency-to-tree-style conversion.
    """
    # Example sentence: "The hungry cat ate a red apple"
    # Dependency structure:
    #   1:The -> 3:cat (det)
    #   2:hungry -> 3:cat (amod)
    #   3:cat -> 4:ate (nsubj)
    #   4:ate -> 0:ROOT (root)
    #   5:a -> 7:apple (det)
    #   6:red -> 7:apple (amod)
    #   7:apple -> 4:ate (obj)
    
    tokens = [
        Token(1, "The", "DET", "DT", 3, "det"),
        Token(2, "hungry", "ADJ", "JJ", 3, "amod"),
        Token(3, "cat", "NOUN", "NN", 4, "nsubj"),
        Token(4, "ate", "VERB", "VBD", 0, "root"),
        Token(5, "a", "DET", "DT", 7, "det"),
        Token(6, "red", "ADJ", "JJ", 7, "amod"),
        Token(7, "apple", "NOUN", "NN", 4, "obj"),
    ]
    
    print("=" * 60)
    print("Original tokens:")
    for t in tokens:
        print(f"  {t.id}: {t.form} -> {t.head} ({t.deprel})")
    
    print("\n" + "=" * 60)
    print("Tree-style representation (with deprel):")
    tree = dependency_to_tree_style(tokens, "XPOS", use_deprel=True)
    tree.pretty_print()
    
    print("\n" + "=" * 60)
    print("Tree-style representation (UAS only):")
    tree_uas = dependency_to_tree_style(tokens, "XPOS", use_deprel=False)
    tree_uas.pretty_print()
    
    print("\n" + "=" * 60)
    print("Arc-standard transitions:")
    transitions = generate_arc_transitions(tokens)
    for i, t in enumerate(transitions):
        print(f"  {i+1}: {t}")
    
    print("\n" + "=" * 60)
    print(f"Verification: {verify_transitions(tokens, transitions)}")
    
    return tree, transitions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
