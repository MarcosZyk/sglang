curl http://10.87.79.111:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
        -d '{
    "model": "qwen/qwen2.5-0.5b-instruct",
        "messages": [
              {"role": "user", "content": "What is the capital of France?"}
                  ]
                    }'
