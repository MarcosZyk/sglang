import pandas as pd
import os
from collections import defaultdict

from tqdm import tqdm

def process_csv(input_file, output_base_dir='output'):
    """
    处理CSV文件并按要求分类存储
    
    Args:
        input_file (str): 输入CSV文件路径
        output_base_dir (str): 输出基础目录
    """
    # 确保输出目录存在
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # 用于存储各分类的数据
    data_by_agent_task_page = defaultdict(list)
    data_by_agent_task = defaultdict(list)
    
    # 分块读取大CSV文件
    chunk_size = 10000  # 可根据内存调整
    print("Scanning the log file...")
    for chunk in tqdm(pd.read_csv(input_file, chunksize=chunk_size)):
        # 按分类存储数据
        for _, row in chunk.iterrows():
            agent_id = str(row['agent_id'])
            task_id = str(row['task_id'])
            page_id = str(row['page_id'])
            
            # 存储到agent/task/page分类
            key_agent_task_page = (agent_id, task_id, page_id)
            data_by_agent_task_page[key_agent_task_page].append(row)
            
            # 存储到agent/task分类（用于overall.csv）
            key_agent_task = (agent_id, task_id)
            data_by_agent_task[key_agent_task].append(row)
    
    print("Writing page log files...")

    # 处理每个page_id的文件
    for (agent_id, task_id, page_id), rows in tqdm(data_by_agent_task_page.items()):
        # 创建目录结构
        dir_path = os.path.join(output_base_dir, agent_id, task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # 转换为DataFrame并按timestamp排序
        df = pd.DataFrame(rows)
        df_sorted = df.sort_values('timestamp')
        
        # 保存到文件
        file_path = os.path.join(dir_path, f"{page_id}.csv")
        df_sorted.to_csv(file_path, index=False)
    
    print("Writing overall log files...")

    # 处理每个agent_id/task_id的overall文件
    for (agent_id, task_id), rows in tqdm(data_by_agent_task.items()):
        # 创建目录结构
        dir_path = os.path.join(output_base_dir, agent_id, task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # 转换为DataFrame并按timestamp排序
        df = pd.DataFrame(rows)
        df_sorted = df.sort_values('timestamp')
        
        # 保存到overall.csv文件
        overall_file_path = os.path.join(dir_path, "overall.csv")
        df_sorted.to_csv(overall_file_path, index=False)

if __name__ == "__main__":
    # 使用示例
    input_csv_file = "input_data.csv"  # 替换为实际的输入文件路径
    process_csv(input_csv_file)