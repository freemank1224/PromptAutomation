from openai import OpenAI
import os
import getpass
import json

# 设置 API 密钥
api_key = os.getenv("OPENAI_API_KEY")  # 从环境变量获取 API 密钥
if not api_key:
    api_key = getpass.getpass("请输入您的 API 密钥: ")

# 创建 OpenAI 客户端，使用自定义端点
client = OpenAI(api_key=api_key, base_url="https://tbnx.plus7.plus/v1")

def chat_with_gpt(prompt, model="gpt-3.5-turbo"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的图像提示词生成助手，擅长创建详细、富有创意的图像描述。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        # 检查响应类型
        if isinstance(response, str):
            # 如果响应是字符串，尝试解析 JSON
            try:
                response_json = json.loads(response)
                return response_json.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            except json.JSONDecodeError:
                return response.strip()
        else:
            # 如果响应是对象，按原方式处理
            return response.choices[0].message.content.strip()
    except Exception as e:
        return f"发生错误: {str(e)}"

# 主循环
while True:
    user_input = input("\n请输入您的图像描述 (输入 'quit' 退出): ")
    if user_input.lower() == 'quit':
        break

    prompt = f"基于以下描述生成一个详细的图像提示词，参考Midjourney的风格：\n\n{user_input}"
    response = chat_with_gpt(prompt)
    print("\n生成的提示词:")
    print(response)

print("程序已退出。")