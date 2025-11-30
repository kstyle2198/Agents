import os

def get_tree(root, indent=""):
    result = ""
    items = sorted(os.listdir(root))
    for i, item in enumerate(items):
        path = os.path.join(root, item)
        is_last = (i == len(items) - 1)
        branch = "└── " if is_last else "├── "
        result += indent + branch + item + "\n"

        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            result += get_tree(path, indent + extension)
    return result

# 사용 예시
tree_str = get_tree(".")
print(tree_str)
