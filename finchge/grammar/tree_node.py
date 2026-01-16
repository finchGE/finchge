import json
from typing import List, Dict, Any, Optional


class TreeNode:
    """
    Represents a node in a parse tree used during genotype-to-phenotype mapping.

    Each node holds a symbol (terminal or non-terminal) and may have children.
    The tree supports JSON and string (CSV-like) serialization.
    Args:
        symbol (str): The symbol associated with this node.
        depth (int, optional): The depth of this node in the tree. Defaults to 0.
    """

    def __init__(self, symbol: str, depth: int = 0):
        self.symbol = symbol
        self.children: List[TreeNode] = []
        self.depth = depth
        self.parent: Optional[TreeNode] = None
        self.root: TreeNode = self
        self.max_depth = 0

    def add_child(self, child: 'TreeNode') -> None:
        """
        Adds a child to the current node and updates depth, parent, and root.

        Args:
            child (TreeNode): Node to add as a child.

        Raises:
            TypeError: If `child` is not a TreeNode instance.
        """
        if not isinstance(child, TreeNode):
            raise TypeError("Child must be a TreeNode instance")

        child.depth = self.depth + 1
        child.parent = self
        # Update root reference for the entire subtree
        self._update_subtree_root(child, self.root)
        self.children.append(child)
        self.root.update_max_depth(child.depth)

    def _update_subtree_root(self, node: 'TreeNode', new_root: 'TreeNode') -> None:
        """
        Recursively updates the root reference for a node and its subtree.

        Args:
            node (TreeNode): Node whose root should be updated.
            new_root (TreeNode): The new root reference.
        """
        node.root = new_root
        for child in node.children:
            self._update_subtree_root(child, new_root)

    def update_max_depth(self, child_depth: int) -> None:
        """
        Updates the tree's maximum depth if the new depth is greater.

        Args:
            child_depth (int): Depth to compare with the current max depth.
        """
        if child_depth > self.root.max_depth:
            self.root.max_depth = child_depth

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the node and its subtree into a dictionary.

        Returns:
            dict: Dictionary representation of the tree.
        """
        return {
            'symbol': self.symbol,
            'depth': self.depth,
            'max_depth': self.max_depth if self.root == self else None,
            'children': [child.to_dict() for child in self.children]
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Serializes the tree into a JSON-formatted string.

        Args:
            indent (int): Indentation for the JSON string.

        Returns:
            str: JSON string representation of the tree.
        """
        try:
            tree_json = json.dumps(self.to_dict(), indent=indent)
        except Exception as e:
            print(e)
        return tree_json

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreeNode':
        """
        Constructs a TreeNode from a dictionary.

        Args:
            data (dict): Dictionary with keys 'symbol', 'depth', and 'children'.

        Returns:
            TreeNode: Reconstructed tree.

        Raises:
            KeyError: If required fields are missing.
        """
        if not all(key in data for key in ['symbol', 'depth', 'children']):
            raise KeyError("Missing required fields in dictionary")

        node = cls(data['symbol'], data['depth'])
        for child_data in data['children']:
            child_node = cls.from_dict(child_data)
            node.add_child(child_node)
        return node

    @classmethod
    def from_json(cls, json_str: str) -> 'TreeNode':
        """
        Constructs a TreeNode from a JSON string.

        Args:
            json_str (str): JSON representation of the tree.

        Returns:
            TreeNode: Reconstructed tree.

        Raises:
            json.JSONDecodeError: If JSON string is invalid.
        """
        try:
            data = json.loads(json_str)
            node = cls.from_dict(data)
            node.root = node
            return node
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON string: {str(e)}", e.doc, e.pos)

    def to_string(self) -> str:
        """
        Serializes the tree into a compact CSV-friendly format.

        Format: (root{child1{grandchild1},child2})

        Returns:
            str: Serialized tree string.
        """
        if not self.children:
            return self.symbol

        children_str = ','.join(child.to_string() for child in self.children)
        return f"({self.symbol}{{{children_str}}})"


    @classmethod
    def from_string(cls, s: str) -> 'TreeNode':
        """
        Reconstructs a TreeNode from a serialized string.

        Args:
            s (str): Serialized string representation.

        Returns:
            TreeNode: Root of the reconstructed tree.
        """

        def parse(s: str, depth: int = 0) -> 'TreeNode':
            if not s.startswith('('):
                return cls(s, depth)

            # Remove outer parentheses
            s = s[1:-1]

            # Get root symbol (everything before first '{')
            symbol_end = s.find('{')
            if symbol_end == -1:
                return cls(s, depth)

            symbol = s[:symbol_end]
            root = cls(symbol, depth)

            # Parse children
            children_str = s[symbol_end + 1:-1]  # Remove { }
            if children_str:
                # Split on top-level commas
                bracket_count = 0
                current = []
                for char in children_str:
                    if char == '{':
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                    elif char == ',' and bracket_count == 0:
                        root.add_child(parse(''.join(current), depth + 1))
                        current = []
                        continue
                    current.append(char)
                if current:
                    root.add_child(parse(''.join(current), depth + 1))

            return root

        return parse(s)


    def __repr__(self, level: int = 0) -> str:
        """
        String representation of the tree node showing its hierarchy.

        Args:
            level: Current indentation level

        Returns:
            String representation of the tree
        """
        indent = "  " * level
        ret = f"{indent}{self.symbol} (depth={self.depth}"
        if self.root == self:
            ret += f", max_depth={self.max_depth}"
        ret += ")\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

    def __eq__(self, other: object) -> bool:
        """
        Compares two TreeNodes for equality.

        Args:
            other: Object to compare with

        Returns:
            True if nodes are equal, False otherwise
        """
        if not isinstance(other, TreeNode):
            return False
        return (self.symbol == other.symbol and
                self.depth == other.depth and
                self.children == other.children)