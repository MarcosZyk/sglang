import os
import pandas as pd
from collections import defaultdict

def analyze_agent_data(base_directory='output'):
    """
    分析agent数据并计算指定的统计信息
    
    Args:
        base_directory (str): 包含分类数据的基础目录路径
    
    Returns:
        dict: 包含每个agent_id的统计信息
    """
    # 存储每个agent_id的统计结果
    results = {}
    
    # 获取所有agent_id目录
    if not os.path.exists(base_directory):
        print(f"目录 {base_directory} 不存在")
        return results
    
    agent_ids = [d for d in os.listdir(base_directory) 
                 if os.path.isdir(os.path.join(base_directory, d))]
    
    # 为每个agent_id计算统计数据
    for agent_id in agent_ids:
        agent_path = os.path.join(base_directory, agent_id)
        if not os.path.isdir(agent_path):
            continue
            
        # 初始化统计结果
        query_count_task0 = 0  # task_id为0中的query记录数
        store_evict_query_count = 0  # 连续store,evict,query序列的数量
        
        # 遍历所有task_id目录
        task_ids = [d for d in os.listdir(agent_path) 
                   if os.path.isdir(os.path.join(agent_path, d))]
        
        for task_id in task_ids:
            task_path = os.path.join(agent_path, task_id)
            if not os.path.isdir(task_path):
                continue
                
            # 获取所有page_id.csv文件（排除overall.csv）
            csv_files = [f for f in os.listdir(task_path) 
                        if f.endswith('.csv') and f != 'overall.csv']
            
            # 处理每个page_id.csv文件
            for csv_file in csv_files:
                csv_path = os.path.join(task_path, csv_file)
                if not os.path.isfile(csv_path):
                    continue
                    
                try:
                    # 读取CSV文件
                    df = pd.read_csv(csv_path)
                    
                    # 检查是否有必要的列
                    required_columns = ['event', 'timestamp']
                    if not all(col in df.columns for col in required_columns):
                        print(f"警告: 文件 {csv_path} 缺少必要的列")
                        continue
                    
                    # 统计1: 如果task_id为0，计算query记录数
                    if task_id == '0':
                        query_events = df[df['event'] == 'query']
                        query_count_task0 += len(query_events)
                    
                    # 统计2: 计算连续的store,evict,query序列数量
                    events = df['event'].tolist()
                    store_evict_query_count += count_store_evict_query_sequences(events)
                    
                except Exception as e:
                    print(f"处理文件 {csv_path} 时出错: {e}")
                    continue
        
        # 存储结果
        results[agent_id] = {
            'query_count_task0': query_count_task0,
            'store_evict_query_sequences': store_evict_query_count
        }
    
    return results

def count_store_evict_query_sequences(events):
    """
    计算events列表中连续的store,evict,query序列数量
    
    Args:
        events (list): 事件列表
    
    Returns:
        int: 序列数量
    """
    count = 0
    i = 0
    while i < len(events) - 2:
        # 检查当前位置是否是store,evict,query序列
        if (events[i] == 'store' and 
            events[i+1] == 'evict' and 
            events[i+2] == 'query'):
            count += 1
            i += 3  # 跳过已匹配的三个元素
        else:
            i += 1  # 移动到下一个元素
    return count

def main():
    """
    主函数
    """
    # 执行分析
    results = analyze_agent_data('output')
    
    # 输出结果
    print("Agent数据分析结果:")
    print("=" * 50)
    for agent_id, stats in results.items():
        print(f"Agent ID: {agent_id}")
        print(f"  Task 0 中的 Query 记录数: {stats['query_count_task0']}")
        print(f"  Store-Evict-Query 序列数: {stats['store_evict_query_sequences']}")
        print("-" * 30)
    
    # 也可以将结果保存到文件
    save_results_to_file(results)

def save_results_to_file(results, filename='analysis_results.txt'):
    """
    将分析结果保存到文件
    
    Args:
        results (dict): 分析结果
        filename (str): 输出文件名
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Agent数据分析结果:\n")
        f.write("=" * 50 + "\n")
        for agent_id, stats in results.items():
            f.write(f"Agent ID: {agent_id}\n")
            f.write(f"  Task 0 中的 Query 记录数: {stats['query_count_task0']}\n")
            f.write(f"  Store-Evict-Query 序列数: {stats['store_evict_query_sequences']}\n")
            f.write("-" * 30 + "\n")
    
    print(f"结果已保存到 {filename}")

if __name__ == "__main__":
    main()