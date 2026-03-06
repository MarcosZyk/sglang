import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv


def analyse_file(file_name):
    prefill_time = 0
    decode_time = 0
    prefill_tokens = 0
    local_cache_tokens = 0
    global_cache_tokens = 0
    with open(file_name, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for num, line in enumerate(csv_reader):
            if(num > 0):
                prefill_time = prefill_time + float(line[2])
                decode_time = decode_time + float(line[3])
                prefill_tokens = prefill_tokens + int(line[4])
    pass

if(__name__ == "__main__"):
    file_lists = os.listdir('../result/')
    result = []
    for file_name in file_lists:
        file = f'../result/{file_name}'
        prefill_time, decode_time, prefill_tokens, local_cache_tokens, global_cache_tokens = analyse_file(file)
        result.append([prefill_time, decode_time, prefill_tokens, local_cache_tokens, global_cache_tokens])
