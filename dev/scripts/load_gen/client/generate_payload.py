import json
import argparse
from typing import Any, Optional

def read_replay_data(file_path: Optional[str] = None, model_name: str = None):
    """
    读取 replay JSON 文件并将其转换为指定格式的字典
    
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
            "task_type_list": [
                "iFlow CLI main",
                "iFlow CLI main",
                "regular task",
                "iFlow CLI main",
            ],
            "semantic_type_list": [
                None,
                ["sub"],
                None,
                ["sub"],
            ],
            "wait_time": [0, 0, 1, 0],
            "tokenizer_name": model_name,
            "openai_model": model_name,
            "temperature": 0.0,
        }

    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化结果字典
    result = {
        "n": len(data),  # JSON文件中的{}个数总和
        "n_task": len({item["task_type_int"] for item in data}),
        "s_list": [],
        "m_list": [],
        "a_list": [],
        "b_list": [],
        "roles_list": [],
        "wait_time": [],
        "temperature": 0.0,
        "openai_model": model_name,
        "tokenizer_name": model_name,
        "task_list": [],
        "task_type_list": [],
        "semantic_type_list": [],
    }
    
    # 遍历数据并填充数组
    for item in data:
        result["s_list"].append(item["s"])
        result["m_list"].append(item["m"])
        result["a_list"].append(item["a"])
        result["b_list"].append(item["b"])
        result["roles_list"].append(item["role"])
        result["wait_time"].append(item["wait_time"])
        result["task_list"].append(item["task_type_int"])
        result["task_type_list"].append(item["task_type"])
        semantic_type: Any = item.get("semantic type")
        if semantic_type is None:
            result["semantic_type_list"].append(None)
        elif isinstance(semantic_type, list):
            result["semantic_type_list"].append(semantic_type)
        else:
            result["semantic_type_list"].append([semantic_type])
    
    return result

if __name__ == "__main__":
    # 使用示例
    result = read_replay_data('../output_json_flatten/test1.json')
    print(result)
