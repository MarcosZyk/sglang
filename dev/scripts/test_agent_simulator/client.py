# client_example.py
import requests
import json

from generate_payload import read_replay_data

URL = "http://127.0.0.1:12306/simulate"

payload = {
    "n": 4,
    "n_task": 2,
    "s_list": [10, 12, 10, 14],
    "m_list": [2, 4, 2, 6],
    "a_list": [[100, 100], [100, 100, 10, 150], [15, 15], [100, 100, 10, 150, 12, 150]],
    "b_list": [
        [0, 0],
        [100, 100, 10, 0],
        [0, 0],
        [
            100,
            100,
            10,
            150,
            12,
            0,
        ],
    ],
    "roles_list": [
        ["system", "user"],
        ["system", "user", "assistant", "user"],
        ["system", "user"],
        ["system", "user", "assistant", "user", "assistant", "user"],
    ],
    "task_list": [0, 0, 1, 0],
    "wait_time": [0, 0, 1, 0],
    "tokenizer_name": "Qwen/Qwen3-8B",
    "openai_model": "Qwen/Qwen3-8B",
    "temperature": 0.0,
}

payload1 = read_replay_data("./replay_data_3.json")

resp = requests.post(URL, json=payload1)
print(resp.status_code)
print(resp.json())
# 若运行后返回 session_id, 可去 sessions/<id>.json 查看完整保存记录
