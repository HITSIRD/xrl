from sklearn.tree import DecisionTreeClassifier
import numpy as np

def tree_to_code(tree, feature_names, output_file="decision_code.txt"):
    """
    将决策树转换为 Python 代码，并写入文件。
    """
    from sklearn.tree import _tree

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    code_lines = []

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            code_lines.append(f"{indent}if features['{name}'] <= {threshold}:")
            code_lines.append(
                f"{indent}    path.append(f\"{feature_names[feature_names.index(name)]} = {{features['{name}']:.3f}} <= {threshold:.3f}\")")
            code_lines.append(f"{indent}    path_index.append({feature_names.index(name)})")
            recurse(tree_.children_left[node], depth + 1)
            code_lines.append(f"{indent}else:")
            code_lines.append(
                f"{indent}    path.append(f\"{feature_names[feature_names.index(name)]} = {{features['{name}']:.3f}} > {threshold:.3f}\")")
            code_lines.append(f"{indent}    path_index.append({feature_names.index(name)})")
            recurse(tree_.children_right[node], depth + 1)
        else:
            output = tree_.value[node].argmax()
            code_lines.append(f"{indent}path.append(\"Skill {output}\")")
            code_lines.append(f"{indent}return {output}")

    code_lines.append("def traced_predict(x, feature_names):")
    code_lines.append("    path = []")
    code_lines.append("    path_index = []")
    code_lines.append("    features = {feature_names[i]: x[i] for i in range(len(feature_names))}")
    code_lines.append("")
    code_lines.append("    def decision_tree(features):")
    recurse(0, 2)
    code_lines.append("    result = decision_tree(features)")
    code_lines.append("    return result, ' -> '.join(path), path_index")

    # 写入文件
    with open(output_file, "w") as file:
        file.write("\n".join(code_lines))


# 动态加载并记录决策路径
def load_decision_program(file_path):
    """
    读取文本文件，加载为 Python 函数，并注入路径跟踪逻辑。
    """
    decision_code = ""
    with open(file_path, "r") as file:
        decision_code = file.read()

    # 动态执行代码，定义函数
    local_namespace = {}
    exec(decision_code, {}, local_namespace)
    return local_namespace["traced_predict"]


# 测试函数
if __name__ == "__main__":
    # 示例数据
    X = np.array([[1, 2, 3], [2, 1, 3], [1, 3, 1], [1, 2, 1], [2, 1, 3], [3, 2, 1]])
    y = np.array([[1], [2], [1], [2], [2], [1]])

    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)

    feature_names = ['Feature A', 'Feature B', 'Feature C']
    # 生成 Python 代码
    tree_to_code(clf, feature_names)

    # 加载动态生成的决策树函数
    traced_predict = load_decision_program("decision_code.txt")

    # 输入特征向量和特征名称
    x = [1, 1, 2]

    # 调用函数并获取结果
    prediction, decision_path = traced_predict(x, feature_names)
    print(f"Prediction: {prediction}")
    print(f"Decision Path: {decision_path}")
