import gradio as gr
from openai import OpenAI
import requests
import logging
from config import config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 预定义的参数模板
PARAMETER_TEMPLATES = {
    "视角": ["ultra wide angle", "close-up", "aerial view", "isometric"],
    "风格": ["photorealistic", "cinematic", "anime style", "oil painting", "comic"],
    "光照": ["soft lighting", "dramatic lighting", "golden hour", "neon lights"],
    "色彩": ["vibrant colors", "muted tones", "monochromatic", "pastel colors"],
    "细节": ["highly detailed", "minimalist", "intricate", "abstract"],
}

def generate_prompt(api_key, description, llm_type, llm_endpoint, llm_model, *args):
    user_params = {}
    for i, (category, options) in enumerate(PARAMETER_TEMPLATES.items()):
        for j, option in enumerate(options):
            checkbox_value = args[i * 8 + j * 2]
            slider_value = args[i * 8 + j * 2 + 1]
            if checkbox_value:
                user_params[option] = slider_value

    prompt = f"基于以下描述和参数生成一个详细的图像提示词，参考Midjourney的风格：\n\n原始描述：{description}\n\n"
    for param, weight in user_params.items():
        prompt += f"参数：{param}，权重：{weight:.1f}\n"
    
    prompt += "\n请生成一个综合考虑原始描述和给定参数的详细提示词，确保权重较高的参数在最终提示词中得到更多体现。"
    prompt += "\n请翻译为英文输出。"

    if llm_type == 'openai':
        client = OpenAI(api_key=api_key, base_url=llm_endpoint)
        
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的图像提示词生成助手，擅长创建详细、富有创意的图像描述。"},
                    {"role": "user", "content": prompt}
                ]
            )
            generated_prompt = response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"API 调用错误：{e}")
            return f"生成提示词时发生错误：{e}\n\n原始描述：{description}"
    
    elif llm_type == 'ollama':
        try:
            response = requests.post(
                f"{llm_endpoint}/api/generate",
                json={
                    "model": llm_model,
                    "prompt": prompt,
                    "system": "你是一个专业的图像提示词生成助手，擅长创建详细、富有创意的图像描述。"
                }
            )
            response.raise_for_status()
            generated_prompt = response.json()['response'].strip()
        except Exception as e:
            logging.error(f"Ollama API 调用错误：{e}")
            return f"生成提示词时发生错误：{e}\n\n原始描述：{description}"
    
    else:
        return f"不支持的 LLM 类型：{llm_type}"

    logging.info(f"成功生成提示词: {generated_prompt[:50]}...")
    return generated_prompt

def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# 提示词生成器")
        api_key_input = gr.Textbox(label="API 密钥", type="password", value=config.OPENAI_API_KEY)
        save_api_key = gr.Checkbox(label="保存 API 密钥")
        
        llm_type = gr.Dropdown(["openai", "ollama"], label="LLM 类型", value=config.LLM_TYPE)
        llm_endpoint = gr.Textbox(label="LLM Endpoint", value=config.LLM_ENDPOINT)
        llm_model = gr.Textbox(label="LLM 模型", value=config.LLM_MODEL)
        
        with gr.Row():
            description_input = gr.Textbox(label="请输入图像描述", lines=3)
        
        parameter_inputs = []
        for category, options in PARAMETER_TEMPLATES.items():
            with gr.Row():
                gr.Markdown(f"## {category}")
            for option in options:
                with gr.Row():
                    checkbox = gr.Checkbox(label=option)
                    slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, label="权重")
                    parameter_inputs.extend([checkbox, slider])

        generate_button = gr.Button("生成提示词")
        output = gr.Textbox(label="生成的提示词", lines=5)

        def save_llm_config(llm_type, endpoint, model):
            config.set_llm_config(llm_type, endpoint, model)
            logging.info("LLM 配置已保存")

        llm_type.change(save_llm_config, inputs=[llm_type, llm_endpoint, llm_model])
        llm_endpoint.change(save_llm_config, inputs=[llm_type, llm_endpoint, llm_model])
        llm_model.change(save_llm_config, inputs=[llm_type, llm_endpoint, llm_model])

        def update_endpoint(llm_type):
            if llm_type == 'openai':
                return config.LLM_ENDPOINT
            elif llm_type == 'ollama':
                return "http://localhost:11434"
            return ""

        llm_type.change(update_endpoint, inputs=[llm_type], outputs=[llm_endpoint])

        generate_button.click(
            generate_prompt,
            inputs=[api_key_input, description_input, llm_type, llm_endpoint, llm_model] + parameter_inputs,
            outputs=output
        )

    return interface

def launch_prompt_generator(description):
    interface = create_interface()
    interface.launch(share=True)  # share=True 允许生成一个公共链接，方便远程访问

# 添加这个部分用于测试和演示
if __name__ == "__main__":
    print("正在启动提示词生成器界面（测试模式）...")
    launch_prompt_generator("这是一个测试描述")