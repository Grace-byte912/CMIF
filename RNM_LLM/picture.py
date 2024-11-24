import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import re


# batch_size=4, seq_len=848, vocab_size=32000
def show_logits(noise_logits, clean_logits, true, batch_size=3, vocab_size=32000):
    pdb.set_trace()
    # 计算概率分布
    noise_probs = torch.softmax(noise_logits, dim=-1)
    clean_probs = torch.softmax(clean_logits, dim=-1)

    # 检查数据范围的函数
    def check_data_range(data, name):
        print(f"logits stats: min={np.min(data)}, max={np.max(data)}, mean={np.mean(data)}, std={np.std(data)}")
        # print(f"{name} Values: {data}")

    # 取对数变换的函数
    def log_transform(data):
        return np.log(data + 1e-10)  # 防止对数零的问题

    seq_index = 0  # 选择要绘制的序列索引
    seq_len = true.shape[1]  # 获取序列的长度（928）

    for token_index in range(seq_len):
        true_label = true[seq_index, token_index].cpu().detach().numpy()  # 获取真实的标签
        if true_label == -100:
            continue
        noise_prob = noise_probs[seq_index, token_index].cpu().detach().numpy()
        clean_prob = clean_probs[seq_index, token_index].cpu().detach().numpy()

        # 检查数据范围
        check_data_range(noise_prob, "Noise Prob")
        check_data_range(clean_prob, "Clean Prob")

        # 取对数变换
        noise_prob_log = log_transform(noise_prob)
        clean_prob_log = log_transform(clean_prob)

        # 绘制概率分布的密度图
        plt.figure(figsize=(10, 6))
        sns.kdeplot(noise_prob_log, color='red', alpha=0.5, label='Noise Logits (Log Transformed)')
        sns.kdeplot(clean_prob_log, color='blue', alpha=0.5, label='Clean Logits (Log Transformed)')

        # 添加真实标签的标记
        plt.axvline(x=np.log(true_label + 1e-10), color='green', linestyle='--',
                    label='True Label (Log Transformed)')

        plt.title(f"Log Transformed Probability Distribution for Token {token_index} in Sequence {seq_index}")
        plt.xlabel("Log Transformed Token ID")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.show()


def show_dialogsum_loss():
    with open('./Log/log_0713_emo_lora4qkvo_dialog_0.4.txt') as file:
        log_data = file.read()
    steps = []
    losses = []
    rouge1s = []
    rouge2s = []
    rougels = []
    acces = []

    start_flag = False
    pattern = re.compile(
        r"train step: (\d+), loss: ([\d.]+), rouge1: ([\d.]+), rouge2: ([\d.]+), rougeL: ([\d.]+), acc: ([\d.]+)"
    )

    for line in log_data.split('\n'):
        if '2024-07-13 15:48:38,081 - INFO - train step: ' in line:
            start_flag = True
            # pdb.set_trace()

        if not start_flag:
            continue

        if start_flag and "train step" in line:
            matches = pattern.findall(line)
            if len(matches) > 0:
                for match in matches:
                    steps.append(int(match[0]))
                    losses.append(float(match[1]))
                    rouge1s.append(float(match[2]))
                    rouge2s.append(float(match[3]))
                    rougels.append(float(match[4]))
                    acces.append(float(match[5]))
        else:
            continue

    plt.figure(figsize=(10, 6))
    # plt.plot(steps, losses, label='Loss')
    plt.plot(steps, rouge1s, label='Rouge1')
    plt.plot(steps, rouge2s, label='Rouge2')
    plt.plot(steps, rougels, label='Rougel')
    # plt.plot(steps, acces, label='Accuracy')
    plt.xlabel('Training Steps')
    plt.ylabel('Metrics')
    plt.title('Training Metrics')
    plt.legend()
    plt.show()



def show_loss():
    # 读取日志文件
    with open('./Log/log_0626_loss_emo_lr6e-7.txt', 'r') as file:
        log_data = file.read()

    steps = []
    losses = []
    bleus = []
    accs = []

    start_flag = False
    for line in log_data.split('\n'):
        if '2024-06-26 09:58:15,351 - INFO - ' in line:
            start_flag = True
            # pdb.set_trace()

        if not start_flag:
            continue

        if start_flag and "train step" in line:
            step = int(re.findall(r'train step: (\d+)', line)[0])
            loss = float(re.findall(r'loss: (\d+\.\d+)', line)[0])
            bleu = float(re.findall(r'BLEU: (\d+\.\d+)', line)[0])
            acc = float(re.findall(r'acc: (\d+\.\d+)', line)[0])

            steps.append(step)
            losses.append(loss)
            bleus.append(bleu)
            accs.append(acc)
        else:
            continue

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Loss')
    plt.plot(steps, bleus, label='BLEU')
    plt.plot(steps, accs, label='Accuracy')
    plt.xlabel('Training Steps')
    plt.ylabel('Metrics')
    plt.title('Training Metrics')
    plt.legend()
    plt.show()

# show_dialogsum_loss()
