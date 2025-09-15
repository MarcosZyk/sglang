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
                                      prompt=test_prompt, temperature=0.0, max_tokens=256)
print(time.time()-start_time, test_prompt, completion.choices[0].text)
