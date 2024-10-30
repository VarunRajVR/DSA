class NaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def delete(self, data_to_delete):
        # Recursively search for the node to delete
        for i, child in enumerate(self.children):
            if child.data == data_to_delete:
                # If the child is the node to delete, remove it
                self.children.pop(i)
                return True
            else:
                # Recursively search in the child nodes
                if child.delete(data_to_delete):
                    return True
        return False

    def update(self, old_data, new_data):
        if self.data == old_data:
            self.data = new_data
            return True
        for child in self.children:
            if child.update(old_data, new_data):
                return True
        return False

    def traverse_preorder(self):
        nodes = [self.data]
        for child in self.children:
            nodes.extend(child.traverse_preorder())
        return nodes

    def traverse_postorder(self):
        nodes = []
        for child in self.children:
            nodes.extend(child.traverse_postorder())
        nodes.append(self.data)
        return nodes

    def traverse_level_order(self):
        nodes = []
        queue = [self]
        
        while queue:
            current_node = queue.pop(0)
            nodes.append(current_node.data)
            queue.extend(current_node.children)
        
        return nodes

# Example Usage:
if __name__ == "__main__":
    # Create the root node
    root = NaryTreeNode("A")

    # Create children of the root
    nodeB = NaryTreeNode("B")
    nodeC = NaryTreeNode("C")
    nodeD = NaryTreeNode("D")

    # Add children to root
    root.add_child(nodeB)
    root.add_child(nodeC)
    root.add_child(nodeD)

    # Create children of nodeB
    nodeE = NaryTreeNode("E")
    nodeF = NaryTreeNode("F")
    nodeB.add_child(nodeE)
    nodeB.add_child(nodeF)

    # Create a child of nodeC
    nodeG = NaryTreeNode("G")
    nodeC.add_child(nodeG)

    print("Initial Tree (Pre-order):", root.traverse_preorder())  # Output: ['A', 'B', 'E', 'F', 'C', 'G', 'D']

    # Update a node
    root.update("E", "X")
    print("Tree after updating 'E' to 'X' (Pre-order):", root.traverse_preorder())  # Output: ['A', 'B', 'X', 'F', 'C', 'G', 'D']

    # Delete a node
    root.delete("C")
    print("Tree after deleting 'C' (Pre-order):", root.traverse_preorder())  # Output: ['A', 'B', 'X', 'F', 'D']
