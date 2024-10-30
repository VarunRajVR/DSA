class AVL():
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.height = 1

    def calculate_height(self):
        left_height = self.left.height if self.left else 0
        right_height = self.right.height if self.right else 0
        self.height = 1 + max(left_height, right_height)

    def calculate_balance_factor(self):
        left_height = self.left.height if self.left else 0
        right_height = self.right.height if self.right else 0
        return left_height - right_height
    
    def right_rotate(self):
        new_root = self.left
        self.left = new_root.right
        new_root.right = self

        # Update heights
        self.calculate_height()
        new_root.calculate_height()

        return new_root

    def left_rotate(self):
        new_root = self.right
        self.right = new_root.left
        new_root.left = self

        # Update heights
        self.calculate_height()
        new_root.calculate_height()

        return new_root

    def left_right_rotate(self):
        self.left = self.left.left_rotate()
        return self.right_rotate()

    def right_left_rotate(self):
        self.right = self.right.right_rotate()
        return self.left_rotate()

    def rebalance(self):
        self.calculate_height()
        balance_factor = self.calculate_balance_factor()

        # Left heavy subtree
        if balance_factor > 1:
            if self.left and self.left.calculate_balance_factor() < 0:
                self.left = self.left.left_rotate()
            return self.right_rotate()

        # Right heavy subtree
        if balance_factor < -1:
            if self.right and self.right.calculate_balance_factor() > 0:
                self.right = self.right.right_rotate()
            return self.left_rotate()

        return self

    def add_child(self, data):
        if data < self.data:
            if self.left:
                self.left = self.left.add_child(data)
            else:
                self.left = AVL(data)
        else:
            if self.right:
                self.right = self.right.add_child(data)
            else:
                self.right = AVL(data)

        # Rebalance the tree
        return self.rebalance()

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
            return self

        if val < self.data:
            if self.left:
                return self.left.search(val)
            else:
                return None

        if val > self.data:
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
            if self.left is None and self.right is None:
                return None

            if self.left is None:
                return self.right
            if self.right is None:
                return self.left

            min_val = self.right.find_min()
            self.data = min_val
            self.right = self.right.delete(min_val)

        # Rebalance the tree after deletion
        return self.rebalance()


def build_tree(elements):
    root = AVL(elements[0])

    for i in range(1, len(elements)):
        root = root.add_child(elements[i])

    return root

if __name__ == '__main__':
    elements = [10, 9, 8,7,6,5,4]
    root = build_tree(elements)
    print(root.in_order_traversal())
