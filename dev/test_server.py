from openai import OpenAI
import time

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:30000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
#test_prompt = "San Francisco is a"
test_prompt = "Please give some more content based on the following content. San Francisco is a  city that has been around for over 100 years. It was founded in 1849 by the Gold Rush, and it's still growing today. The city is home to many different neighborhoods and districts, each with their own unique character and culture. One of the most popular neighborhoods in San Francisco is Chinatown. This neighborhood is known for its vibrant Chinese-American community, as well as its delicious cuisine and cultural events. If you're visiting San Francisco during your trip, don't miss out on exploring this beloved neighborhood!"
start_time=time.time()
completion = client.completions.create(model="deepseek-ai/DeepSeek-V2-Lite-Chat",
                                      prompt=test_prompt, temperature=0.0, max_tokens=2048)
# while True:
print("========Question=======")
print(test_prompt)
print("=========Answer========")
print(time.time()-start_time)
print(completion.choices[0].text)

response = 0
dialect = test_prompt

follow_up = [
    'Please give me more descriptions about its cultural.',
    'Please give me more descriptions about its style of architecture.',
    'Please give me more descriptions about the living costs in this city.',
    'Please give me more descriptions about the famous people from this city.',
]
idx = 0

while response < 4 * 1024:

    start_time = time.time()
    completion = client.completions.create(model="deepseek-ai/DeepSeek-V2-Lite-Chat",
                                           prompt=dialect, temperature=0.0, max_tokens=8192)
    print(time.time() - start_time)
    print(completion.usage.total_tokens)
    print(completion.choices[0].text)

    response = completion.usage.total_tokens
    dialect += dialect + '\nHere\'the answer. ' + completion.choices[0].text + '\nAll above are previous questions and answers. ' + follow_up[idx]
    idx += 1

    if idx == len(follow_up):
        break




