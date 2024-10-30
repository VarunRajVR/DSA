class BinarySearchTreeNode():
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def add_child(self, data):
        if data == self.data:
            return

        if data < self.data:
            # add data in left subtree
            if self.left:
                self.left.add_child(data)
            else:
                self.left = BinarySearchTreeNode(data)
        else:
            if self.right:
                self.right.add_child(data)
            else:
                self.right = BinarySearchTreeNode(data)

    def in_order_traversal(self):
        elements = []

        # Visit left subtree
        if self.left:
            elements += self.left.in_order_traversal()

        # Visit base node (current node)
        elements.append(self.data)

        # Visit right subtree
        if self.right:
            elements += self.right.in_order_traversal()

        return elements

    def search(self, val):
        if self.data == val:
            # The value is found, return the current node
            return self

        if val < self.data:
            # Value might be in the left subtree
            if self.left:
                return self.left.search(val)
            else:
                return None

        if val > self.data:
            # Value might be in the right subtree
            if self.right:
                return self.right.search(val)
            else:
                return None
            
    def find_max(self):
    
        if self.right is None:
            return self.data
        return self.right.find_max()
    
    def find_min(self):
        if self.left is None:
            return self.data
        return self.left.find_min()
    
    def delete(self, val):
        if val < self.data:
            if self.left:
                self.left = self.left.delete(val)
        elif val > self.data:
            if self.right:
                self.right = self.right.delete(val)
        else:
            # Case 1: Node with no children
            if self.left is None and self.right is None:
                return None

            # Case 2: Node with one child
            if self.left is None:
                return self.right
            if self.right is None:
                return self.left

            # Case 3: Node with two children
            min_val = self.right.find_min()
            self.data = min_val
            self.right = self.right.delete(min_val)

        return self



def build_tree(elements):
    root = BinarySearchTreeNode(elements[0])

    for i in range(1, len(elements)):
        root.add_child(elements[i])

    return root

if __name__ =='__main__':
    elements = [2,3,4,2,4,11,77,12,55,15]
    root= build_tree(elements)
    print(root.in_order_traversal())

