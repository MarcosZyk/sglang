import json
import argparse
from typing import Optional

def read_replay_data(file_path: Optional[str] = None):
    """
    读取 replay_data_3.json 文件并将其转换为指定格式的字典
    
    参数:
    file_path: JSON文件路径
    
    返回:
    dict: 包含提取数据的字典
    """

    if(file_path == None):
        return  {
            "n": 4,
            "n_task": 2,
            "s_list": [10, 12, 10, 14],
            "m_list": [2, 4, 2, 6],
            "a_list": [[100, 100], [100, 100, 10, 150], [15, 15], [100, 100, 10, 150, 12, 150]],
            "b_list": [[0, 0], [100, 100, 10, 0], [0, 0], [100, 100, 10, 150, 12, 0,]],
            "roles_list": [
                ["system", "user"],
                ["system", "user", "assistant", "user"],
                ["system", "user"],
                ["system", "user", "assistant", "user", "assistant", "user"],
            ],
            "task_list": [0, 0, 1, 0],
            "wait_time": [0, 0, 1, 0],
            "tokenizer_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "openai_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "temperature": 0.0,
        }

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct', type=str)
    args = parser.parse_args()
    model_name = args.model_name

    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化结果字典
    result = {
        "n": len(data),  # JSON文件中的{}个数总和
        "n_task": len(set(item["task_list"] for item in data)),  # task_list中的数值种类总数
        "s_list": [],
        "m_list": [],
        "a_list": [],
        "b_list": [],
        "role_list": [],
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
        result["role_list"].append(item["role_list"])
        result["wait_time"].append(item["wait_time"])
        result["task_list"].append(item["task_list"])
    
    return result

if __name__ == "__main__":
    # 使用示例
    result = read_replay_data('../dataset/replay_data_3.json')
    print(result)