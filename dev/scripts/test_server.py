from email.policy import default

from openai import OpenAI
import time
import argparse


def do_conversation(client, model, init_prompt: str):
    # while True:
    response = 0
    dialect = init_prompt

    follow_up = [
        'Please give me more descriptions about its cultural.',
        'Please give me more descriptions about its style of architecture.',
        'Please give me more descriptions about the living costs in this city.',
        'Please give me more descriptions about the famous people from this city.',
    ]
    idx = 0

    while response < 2048:

        start_time = time.time()
        completion = client.completions.create(model=model,prompt=dialect, temperature=0.0, max_tokens=2048)
        print(time.time() - start_time)
        print(completion.usage.total_tokens)
        print(completion.choices[0].text)

        response = completion.usage.total_tokens
        dialect += dialect + '\nHere\'the answer. ' + completion.choices[
            0].text + '\nAll above are previous questions and answers. ' + follow_up[idx]
        idx += 1

        if idx == len(follow_up):
            break


def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description='这是一个示例程序')

    # 添加参数
    parser.add_argument('--ip', '-i', type=str, default="127.0.0.1", )
    parser.add_argument('--port', '-p', type=int, default=30000, )
    parser.add_argument('--model', '-m', type=str, default="Qwen/Qwen3-8B", )

    # 解析参数
    args = parser.parse_args()

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{args.ip}:{args.port}/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    # test_prompt = "San Francisco is a"
    prompt_1 = "Please give some more content based on the following content. Do not think too much time! San Francisco is a  city that has been around for over 100 years. It was founded in 1849 by the Gold Rush, and it's still growing today. The city is home to many different neighborhoods and districts, each with their own unique character and culture. One of the most popular neighborhoods in San Francisco is Chinatown. This neighborhood is known for its vibrant Chinese-American community, as well as its delicious cuisine and cultural events. If you're visiting San Francisco during your trip, don't miss out on exploring this beloved neighborhood!"
    do_conversation(client, args.model, prompt_1)

    prompt_2 = "Please give some more content based on the following content. Do not think too much time! New York is a city that has been around for over 100 years. The city is home to many different neighborhoods and districts, each with their own unique character and culture. One of the most popular neighborhoods in San Francisco is Chinatown. This neighborhood is known for its vibrant Chinese-American community, as well as its delicious cuisine and cultural events."
    do_conversation(client, args.model, prompt_2)

if __name__ == "__main__":
    main()
