"""
Defines arc-standard transition operations for dependency parsing.
Arc-standard 依存转移操作定义

Transitions for arc-standard dependency parsing:
- SHIFT: Push the next word from buffer onto stack
- LEFTARC: Create a left arc (stack[-2] <- stack[-1]), remove stack[-2]
- RIGHTARC: Create a right arc (stack[-2] -> stack[-1]), remove stack[-1]

Optionally with dependency relation labels:
- LEFTARC(label): Left arc with specific relation
- RIGHTARC(label): Right arc with specific relation
"""

from abc import ABC, abstractmethod
import ast
from enum import Enum
import functools
import logging

logger = logging.getLogger('stanza')


class DependencyTransitionScheme(Enum):
    """
    Dependency parsing transition schemes.
    依存分析转移方案
    """
    def __new__(cls, value, short_name):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.short_name = short_name
        return obj

    # Arc-standard: SHIFT, LEFTARC, RIGHTARC
    # Most common bottom-up transition system
    ARC_STANDARD = 1, "arcstd"

    # Arc-eager: SHIFT, LEFTARC, RIGHTARC, REDUCE
    # Can build some arcs earlier than arc-standard
    # ARC_EAGER = 2, "arceag"  # Future work


@functools.total_ordering
class DepTransition(ABC):
    """
    Abstract base class for dependency transitions.
    依存转移操作的抽象基类
    
    Each transition modifies:
    - stack: List of partial trees being built
    - buffer: Remaining words to process
    - arcs: Set of dependency arcs created
    """
    
    @abstractmethod
    def update_state(self, state, model):
        """
        Apply this transition to the state.
        将此转移应用到状态
        
        Returns tuple:
        - updated buffer position
        - updated stack
        - new constituent/tree node (if any), and builder method
        """
        pass
    
    @abstractmethod
    def is_legal(self, state, model):
        """
        Check if this transition is legal in current state.
        检查此转移在当前状态是否合法
        """
        pass
    
    @abstractmethod
    def short_name(self):
        """
        A short name to identify this transition.
        此转移的简短名称
        """
        pass
    
    def short_label(self):
        if not hasattr(self, "label"):
            return self.short_name()
        if self.label is None:
            return self.short_name()
        return "{}({})".format(self.short_name(), self.label)
    
    def apply(self, state, model):
        """
        Return a new State transformed via this transition.
        返回通过此转移变换后的新状态
        """
        update = model.bulk_apply([state], [self])
        return update[0]
    
    def components(self):
        """Return components of this transition (for compound transitions)."""
        return [self]
    
    def __lt__(self, other):
        # Shift comes first, then alphabetically
        if self == other:
            return False
        if isinstance(self, Shift):
            return True
        if isinstance(other, Shift):
            return False
        return str(self) < str(other)
    
    @staticmethod
    def from_repr(desc):
        """
        Deserialize a transition from its string representation.
        从字符串表示反序列化转移操作
        
        Avoids using eval() for security.
        """
        if desc == 'Shift':
            return Shift()
        if desc == 'LeftArc':
            return LeftArc()
        if desc == 'RightArc':
            return RightArc()
        
        # Parse labeled arcs: LeftArc(nsubj) or RightArc(dobj)
        if desc.startswith('LeftArc(') and desc.endswith(')'):
            label = desc[8:-1]  # Extract label between parentheses
            return LeftArc(label)
        if desc.startswith('RightArc(') and desc.endswith(')'):
            label = desc[9:-1]
            return RightArc(label)
        
        raise ValueError(f"Unknown Transition: {desc}")


class Shift(DepTransition):
    """
    SHIFT: Move the first word from buffer to stack.
    SHIFT: 将 buffer 第一个词移到 stack
    
    Precondition: Buffer is not empty
    前置条件：Buffer 非空
    """
    
    def update_state(self, state, model):
        """
        Push buffer[0] onto stack, advance buffer position.
        将 buffer[0] 压入栈，buffer 位置前进
        """
        # Transform word to a constituent/tree node
        new_constituent = model.transform_word_to_constituent(state)
        return state.word_position + 1, state.constituents, new_constituent, None
    
    def is_legal(self, state, model):
        """
        Legal if buffer is not empty.
        Buffer 非空时合法
        
        For arc-standard:
        - Cannot shift if buffer is empty
        - At the end, need at least one item on stack to reduce
        """
        if state.empty_word_queue():
            return False
        return True
    
    def short_name(self):
        return "Shift"
    
    def __repr__(self):
        return "Shift"
    
    def __eq__(self, other):
        if self is other:
            return True
        return isinstance(other, Shift)
    
    def __hash__(self):
        return hash(37)


class LeftArc(DepTransition):
    """
    LEFTARC: Create arc from stack[-1] to stack[-2], remove stack[-2].
    LEFTARC: 从 stack[-1] 指向 stack[-2] 建立弧，移除 stack[-2]
    
    In tree representation: stack[-2] becomes child of stack[-1]
    树形表示：stack[-2] 成为 stack[-1] 的子节点
    
    stack[-2].head = stack[-1]
    
    Precondition: 
    - Stack has at least 2 elements
    - stack[-2] is not ROOT (id=0)
    前置条件：
    - 栈至少有2个元素
    - stack[-2] 不是 ROOT (id=0)
    """
    
    def __init__(self, label=None):
        """
        Initialize LeftArc, optionally with a dependency relation label.
        初始化 LeftArc，可选依存关系标签
        
        Args:
            label: Dependency relation (e.g., 'nsubj', 'dobj', 'amod')
                   If None, only UAS is considered (no labeled arcs)
        """
        self.label = label
    
    def update_state(self, state, model):
        """
        Pop stack[-2], make it child of stack[-1].
        弹出 stack[-2]，使其成为 stack[-1] 的子节点
        
        Stack: [..., A, B] -> [..., B_with_A_as_child]
        """
        constituents = state.constituents
        
        # Get top two elements: stack[-1] (head) and stack[-2] (dependent)
        head = constituents.value  # stack[-1], will be the head
        constituents = constituents.pop()
        dependent = constituents.value  # stack[-2], will become child
        constituents = constituents.pop()
        
        # Build new tree: head with dependent as left child
        # The label is the POS tag of the head (for tree representation)
        # Children order: [dependent, head's original children...]
        children = [dependent]
        
        return state.word_position, constituents, (self.label, head, children), LeftArc
    
    @staticmethod
    def build_constituents(model, data):
        """
        Build new constituents for LeftArc operations.
        为 LeftArc 操作构建新的成分
        
        data: list of (label, head, children) tuples
        """
        results = []
        for label, head, children in data:
            # Create tree where head dominates dependent
            new_tree = model.build_dependency_node(head, children, 'left', label)
            results.append(new_tree)
        return results
    
    def is_legal(self, state, model):
        """
        Legal if stack has >= 2 elements and stack[-2] is not ROOT.
        栈有>=2个元素且 stack[-2] 不是 ROOT 时合法
        """
        # Need at least 2 items on stack
        if state.num_constituents() < 2:
            return False
        
        # Cannot make ROOT a dependent (stack[-2] shouldn't be ROOT)
        # In practice, we track this via the word index
        # ROOT is typically index 0 or represented specially
        constituents = state.constituents
        # Skip top element to get stack[-2]
        second = constituents.pop().value if constituents.pop() else None
        if second is not None:
            # Check if it's the artificial ROOT node
            if hasattr(second, 'is_root') and second.is_root():
                return False
        
        return True
    
    def short_name(self):
        return "LA" if self.label is None else f"LA"
    
    def __repr__(self):
        if self.label is None:
            return "LeftArc"
        return f"LeftArc({self.label})"
    
    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, LeftArc):
            return False
        return self.label == other.label
    
    def __hash__(self):
        return hash(('LeftArc', self.label))


class RightArc(DepTransition):
    """
    RIGHTARC: Create arc from stack[-2] to stack[-1], remove stack[-1].
    RIGHTARC: 从 stack[-2] 指向 stack[-1] 建立弧，移除 stack[-1]
    
    In tree representation: stack[-1] becomes child of stack[-2]
    树形表示：stack[-1] 成为 stack[-2] 的子节点
    
    stack[-1].head = stack[-2]
    
    Precondition: Stack has at least 2 elements
    前置条件：栈至少有2个元素
    """
    
    def __init__(self, label=None):
        """
        Initialize RightArc, optionally with a dependency relation label.
        初始化 RightArc，可选依存关系标签
        
        Args:
            label: Dependency relation (e.g., 'nsubj', 'dobj', 'amod')
                   If None, only UAS is considered
        """
        self.label = label
    
    def update_state(self, state, model):
        """
        Pop stack[-1], make it child of stack[-2].
        弹出 stack[-1]，使其成为 stack[-2] 的子节点
        
        Stack: [..., A, B] -> [..., A_with_B_as_child]
        """
        constituents = state.constituents
        
        # Get top two elements: stack[-1] (dependent) and stack[-2] (head)
        dependent = constituents.value  # stack[-1], will become child
        constituents = constituents.pop()
        head = constituents.value  # stack[-2], will be the head
        constituents = constituents.pop()
        
        # Build new tree: head with dependent as right child
        children = [dependent]
        
        return state.word_position, constituents, (self.label, head, children), RightArc
    
    @staticmethod
    def build_constituents(model, data):
        """
        Build new constituents for RightArc operations.
        为 RightArc 操作构建新的成分
        
        data: list of (label, head, children) tuples
        """
        results = []
        for label, head, children in data:
            # Create tree where head dominates dependent
            new_tree = model.build_dependency_node(head, children, 'right', label)
            results.append(new_tree)
        return results
    
    def is_legal(self, state, model):
        """
        Legal if stack has >= 2 elements.
        栈有>=2个元素时合法
        
        Additional constraint: 
        - If buffer is empty and stack has exactly 2 elements,
          must do RightArc to finish (the remaining element connects to ROOT)
        """
        # Need at least 2 items on stack
        if state.num_constituents() < 2:
            return False
        return True
    
    def short_name(self):
        return "RA" if self.label is None else f"RA"
    
    def __repr__(self):
        if self.label is None:
            return "RightArc"
        return f"RightArc({self.label})"
    
    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, RightArc):
            return False
        return self.label == other.label
    
    def __hash__(self):
        return hash(('RightArc', self.label))


class Finalize(DepTransition):
    """
    FINALIZE: Mark parsing as complete, connect remaining element to ROOT.
    FINALIZE: 标记解析完成，将剩余元素连接到 ROOT
    
    This transition is applied when:
    - Buffer is empty
    - Stack has exactly 1 element (the complete dependency tree)
    
    在以下情况应用此转移：
    - Buffer 为空
    - Stack 恰好有1个元素（完整的依存树）
    """
    
    def __init__(self, root_label='ROOT'):
        self.label = root_label
    
    def update_state(self, state, model):
        """
        Wrap the final tree with ROOT.
        用 ROOT 包装最终的树
        """
        constituents = state.constituents
        final_tree = constituents.value
        constituents = constituents.pop()
        
        # Create ROOT node wrapping the tree
        return state.word_position, constituents, (self.label, [final_tree]), Finalize
    
    @staticmethod
    def build_constituents(model, data):
        """Build ROOT wrapper for final tree."""
        results = []
        for label, children in data:
            new_tree = model.build_root_node(label, children)
            results.append(new_tree)
        return results
    
    def is_legal(self, state, model):
        """
        Legal only when buffer empty and stack has exactly 1 element.
        仅当 buffer 为空且 stack 恰好有1个元素时合法
        """
        return state.empty_word_queue() and state.has_one_constituent()
    
    def short_name(self):
        return "Fin"
    
    def __repr__(self):
        return f"Finalize({self.label})"
    
    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Finalize):
            return False
        return self.label == other.label
    
    def __hash__(self):
        return hash(('Finalize', self.label))


def get_all_transitions(include_labels=False, labels=None):
    """
    Get all possible transitions for dependency parsing.
    获取所有可能的依存分析转移操作
    
    Args:
        include_labels: If True, include labeled arc transitions
        labels: List of dependency relation labels (required if include_labels=True)
    
    Returns:
        List of Transition objects
    """
    transitions = [Shift()]
    
    if include_labels and labels:
        for label in labels:
            transitions.append(LeftArc(label))
            transitions.append(RightArc(label))
    else:
        transitions.append(LeftArc())
        transitions.append(RightArc())
    
    transitions.append(Finalize())
    
    return transitions


def check_transitions(train_transitions, other_transitions, treebank_name):
    """
    Check that all transitions in other dataset exist in train set.
    检查其他数据集中的所有转移是否存在于训练集中
    """
    unknown = set()
    for trans in other_transitions:
        if trans not in train_transitions:
            for component in trans.components():
                if component not in train_transitions:
                    raise RuntimeError(
                        f"Found transition {trans} in {treebank_name} "
                        f"which doesn't exist in train set"
                    )
            unknown.add(trans)
    
    if unknown:
        logger.warning(
            "Found transitions with known components but unknown combination: %s",
            sorted(unknown)
        )


# ============================================================
# Oracle Functions for Arc-Standard
# Arc-Standard 的 Oracle 函数
# ============================================================

def arc_standard_oracle(sentence):
    """
    Static oracle for arc-standard dependency parsing.
    Arc-standard 依存分析的静态 oracle
    
    Given a sentence with gold dependency annotations,
    returns the sequence of transitions to build the tree.
    给定带有金标依存标注的句子，
    返回构建依存树的转移序列。
    
    Args:
        sentence: List of tokens with 'id', 'head', 'deprel' fields
                 Token 列表，包含 'id', 'head', 'deprel' 字段
    
    Returns:
        List of Transition objects
        转移操作对象列表
    
    Note: This assumes the sentence is projective.
    注意：假设句子是投射性的。
    """
    # Build head lookup: token_id -> head_id
    heads = {}
    deprels = {}
    for token in sentence:
        token_id = int(token['id'])
        head_id = int(token['head'])
        heads[token_id] = head_id
        deprels[token_id] = token.get('deprel', None)
    
    n = len(sentence)
    stack = [0]  # Stack starts with ROOT (id=0)
    buffer = list(range(1, n + 1))  # Buffer has word ids 1..n
    transitions = []
    
    while buffer or len(stack) > 1:
        if len(stack) >= 2:
            s1 = stack[-1]  # Top of stack
            s2 = stack[-2]  # Second on stack
            
            # Check for LeftArc: s2 <- s1 (s1 is head of s2)
            if heads.get(s2, -1) == s1:
                # Check that s2 has no dependents remaining in buffer
                has_deps = any(heads.get(b, -1) == s2 for b in buffer)
                if not has_deps:
                    label = deprels.get(s2)
                    transitions.append(LeftArc(label))
                    stack.pop(-2)  # Remove s2
                    continue
            
            # Check for RightArc: s2 -> s1 (s2 is head of s1)
            if heads.get(s1, -1) == s2:
                # Check that s1 has no dependents remaining in buffer or stack
                has_deps_buffer = any(heads.get(b, -1) == s1 for b in buffer)
                has_deps_stack = any(heads.get(s, -1) == s1 for s in stack[:-1])
                if not has_deps_buffer and not has_deps_stack:
                    label = deprels.get(s1)
                    transitions.append(RightArc(label))
                    stack.pop()  # Remove s1
                    continue
        
        # Default: Shift
        if buffer:
            transitions.append(Shift())
            stack.append(buffer.pop(0))
        else:
            # Should not reach here for projective trees
            raise ValueError("Cannot complete parse - is the tree projective?")
    
    # Add Finalize
    transitions.append(Finalize())
    
    return transitions


def transitions_to_string(transitions):
    """
    Convert transition sequence to string representation.
    将转移序列转换为字符串表示
    """
    return ' '.join(t.short_label() for t in transitions)


def string_to_transitions(s, include_labels=False):
    """
    Parse string representation back to transitions.
    将字符串表示解析回转移序列
    """
    parts = s.strip().split()
    transitions = []
    for part in parts:
        transitions.append(DepTransition.from_repr(part))
    return transitions


# ============================================================
# Test / Demo
# 测试 / 演示
# ============================================================

if __name__ == '__main__':
    # Simple test
    print("=== Dependency Transition Classes ===")
    print("依存转移类")
    print()
    
    # Create transitions
    shift = Shift()
    left_arc = LeftArc()
    right_arc = RightArc()
    left_arc_nsubj = LeftArc('nsubj')
    right_arc_dobj = RightArc('dobj')
    finalize = Finalize()
    
    print("Transitions / 转移操作:")
    print(f"  {shift} -> short: {shift.short_name()}")
    print(f"  {left_arc} -> short: {left_arc.short_name()}")
    print(f"  {right_arc} -> short: {right_arc.short_name()}")
    print(f"  {left_arc_nsubj} -> short: {left_arc_nsubj.short_label()}")
    print(f"  {right_arc_dobj} -> short: {right_arc_dobj.short_label()}")
    print(f"  {finalize} -> short: {finalize.short_name()}")
    print()
    
    # Test from_repr
    print("Deserialization / 反序列化:")
    for desc in ['Shift', 'LeftArc', 'RightArc', 'LeftArc(nsubj)', 'RightArc(dobj)']:
        t = DepTransition.from_repr(desc)
        print(f"  '{desc}' -> {t}")
    print()
    
    # Test oracle on simple sentence
    print("=== Oracle Test / Oracle 测试 ===")
    # "I saw her" with dependencies:
    # I(1) -> saw(2) [nsubj]
    # saw(2) -> ROOT(0) [root]  
    # her(3) -> saw(2) [dobj]
    sentence = [
        {'id': '1', 'form': 'I', 'head': '2', 'deprel': 'nsubj'},
        {'id': '2', 'form': 'saw', 'head': '0', 'deprel': 'root'},
        {'id': '3', 'form': 'her', 'head': '2', 'deprel': 'dobj'},
    ]
    
    print("Sentence: I saw her")
    print("Dependencies: I->saw(nsubj), saw->ROOT(root), her->saw(dobj)")
    print()
    
    try:
        oracle_transitions = arc_standard_oracle(sentence)
        print("Oracle transitions / Oracle 转移序列:")
        for i, t in enumerate(oracle_transitions):
            print(f"  {i+1}. {t}")
    except Exception as e:
        print(f"Oracle error: {e}")
