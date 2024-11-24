import pdb

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


def parse_file(file_path):
    predictions = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Output:"):
                # 提取 Output 和 Label
                # pdb.set_trace()
                output = line.split("This movie review is")[-1]
                if "positive" in output.lower():
                    predictions.append("positive")
                elif "negative" in output.lower():
                    predictions.append("negative")
                else:
                    predictions.append("negative")
                    print("ERRor")

            elif line.startswith("Label:"):
                label = line.split("Label:")[-1]
                # 处理可能的 "positive" 或 "negative" 情况
                if "positive" in label:
                    labels.append("positive")
                elif "negative" in label:
                    labels.append("negative")
                else:
                    labels.append("negative")
                    print("ERRor")
            else:
                print("ERRor")

    return predictions, labels


def calculate_metrics(predictions, labels):
    # 使用 sklearn 计算指标
    precision = precision_score(labels, predictions, pos_label="positive", average='binary')
    recall = recall_score(labels, predictions, pos_label="positive", average='binary')
    f1 = f1_score(labels, predictions, pos_label="positive", average='binary')

    # 打印分类报告
    print("Classification Report:")
    print(classification_report(labels, predictions, target_names=["negative", "positive"]))

    # 返回具体指标
    return precision, recall, f1


if __name__ == "__main__":
    # 文件路径
    file_path = "Llama2_sst2_2.0.txt"  # 替换为你的文件路径

    # 解析文件
    predictions, labels = parse_file(file_path)

    # 计算并打印指标
    precision, recall, f1 = calculate_metrics(predictions, labels)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")