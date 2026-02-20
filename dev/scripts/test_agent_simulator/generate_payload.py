import json
import argparse

def read_replay_data(file_path):
    """
    读取 replay_data_3.json 文件并将其转换为指定格式的字典

    参数:
    file_path: JSON文件路径

    返回:
    dict: 包含提取数据的字典
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-8B', type=str)
    args = parser.parse_args()
    model_name = args.model

    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化结果字典
    result = {
        "n": len(data),  # JSON文件中的{}个数总和
        "n_task": 2, #len(set(item["task_list"] for item in data)),  # task_list中的数值种类总数
        "s_list": [],
        "m_list": [],
        "a_list": [],
        "b_list": [],
        "roles_list": [],
        "wait_time": [],
        "tempurature": 0,
        "openai_model": model_name,
        "tokenizer_name": model_name,
        "task_list": []
    }

    # 遍历数据并填充数组
    for item in data:
        result["s_list"].append(item["s_list"])
        result["m_list"].append(item["m_list"])
        result["a_list"].append(item["a_list"])
        result["b_list"].append(item["b_list"])
        result["roles_list"].append(item["role_list"])
        result["wait_time"].append(item["wait_time"])
        result["task_list"].append(item["task_list"])

    return result

# 使用示例
if __name__ == "__main__":
    result = read_replay_data('./replay_data_3.json')
    print(result)
