import gradio as gr
from openai import OpenAI
import requests
import logging
import os
from PIL import Image
from config import config
from comfyui_api import comfyui_api

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
    logging.info(f"LLM Type: {llm_type}")
    logging.info(f"LLM Endpoint: {llm_endpoint}")
    logging.info(f"LLM Model: {llm_model}")
    logging.info(f"===============")

    user_params = {}
    for i, options in enumerate(PARAMETER_TEMPLATES.values()):
        for j, option in enumerate(options):
            checkbox_value = args[i * 8 + j * 2]
            slider_value = args[i * 8 + j * 2 + 1]
            if checkbox_value and slider_value > 0:
                user_params[option] = int(slider_value)  # 确保权重为整数

    prompt = f"基于以下描述生成一个详细的图像提示词，参考Midjourney的风格：\n\n原始描述：{description}\n\n"
    prompt += "如果用户的描述只是简单的几个关键词，请加入合理想象，利用给定关键词生成一个完整的场景描述。"
    prompt += "请生成详细的提示词，不要包含任何额外的参数或权重信息。"
    prompt += "\n请翻译为英文，并且仅输出英文提示词。"

    if llm_type == 'openai':
        llm_endpoint = config.OPENAI_ENDPOINT
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
        llm_endpoint = config.OLLAMA_ENDPOINT
        client = OpenAI(api_key='ollama', base_url=llm_endpoint)
        try:
            full_url = llm_endpoint
            logging.info(f"Calling Ollama API at: {full_url}")
            response = client.chat.completions.create(
                model="llama3.1",
                messages=[
                    {"role": "system", "content": "你是一个专业的图像提示词生成助手，擅长创建详细、富有创意的图像描述。"},
                    {"role": "user", "content": prompt}
                ]
            )
            generated_prompt = response.choices[0].message.content.strip()
            logging.info(f"Ollama API 调用成功：{generated_prompt}")

        except Exception as e:
            logging.error(f"Ollama API 调用错误：{e}")
            return f"生成提示词时发生错误：{e}\n\n原始描述：{description}"
    
    else:
        return f"不支持的 LLM 类型：{llm_type}"

    # 在生成的提示词后面添加参数和权重，只包含权重不为 0 的参数
    for param, weight in user_params.items():
        param_formatted = param.lower().replace(" ", "_")
        generated_prompt += f" {param_formatted}::{weight}，"

    logging.info(f"成功生成提示词: {generated_prompt[:50]}...")
    return generated_prompt

def generate_image(prompt, workflow_name):
    try:
        image_filename = comfyui_api.generate_image(prompt, workflow_name)
        if image_filename:
            # 构造完整的文件路径
            image_path = os.path.join(config.OUTPUT_DIR, image_filename)
            # 检查文件是否存在
            if os.path.exists(image_path):
                # 使用 PIL 打开图像并返回
                return Image.open(image_path)
            else:
                logging.error(f"生成的图像文件不存在: {image_path}")
                return None
        else:
            return None
    except Exception as e:
        logging.error(f"图像生成错误: {e}")
        return None

def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# 提示词生成器和图像生成器")
        
        with gr.Row():
            # 左侧栏
            with gr.Column(scale=1):
                api_key_input = gr.Textbox(label="API 密钥", type="password", value=config.OPENAI_API_KEY)
                save_api_key = gr.Checkbox(label="保存 API 密钥")
                
                llm_type = gr.Dropdown(["openai", "ollama"], label="LLM 类型", value=config.LLM_TYPE)
                
                # OpenAI 设置
                with gr.Group(visible=config.LLM_TYPE == "openai") as openai_group:
                    openai_endpoint = gr.Textbox(label="OpenAI Endpoint", value=config.OPENAI_ENDPOINT, type="text")
                    openai_model = gr.Dropdown(
                        ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"],
                        label="OpenAI 模型",
                        value=config.OPENAI_MODEL
                    )
                
                # Ollama 设置
                with gr.Group(visible=config.LLM_TYPE == "ollama") as ollama_group:
                    ollama_endpoint = gr.Textbox(label="Ollama Endpoint", value=config.OLLAMA_ENDPOINT, type="text")
                    print(ollama_endpoint.value)
                    
                description_input = gr.Textbox(label="请输入图像描述", lines=5)
                generate_button = gr.Button("生成提示词", variant="primary")

                with gr.Row():
                    workflows = comfyui_api.list_workflows()
                    workflow_dropdown = gr.Dropdown(choices=workflows, label="选择工作流")
                    generate_image_button = gr.Button("生成图像")
            
            # 右侧栏（用于显示生成的提示词和图像）
            with gr.Column(scale=1):
                with gr.Row():
                    output = gr.Textbox(label="生成的提示词", lines=10)

                with gr.Row():
                    output_image = gr.Image(label="生成的图像", type="pil")

        # 下方参数选择区域（分为三列）
        with gr.Row():
            parameter_inputs = []
            for i, (category, options) in enumerate(PARAMETER_TEMPLATES.items()):
                with gr.Column(scale=1):
                    gr.Markdown(f"## {category}")
                    for option in options:
                        with gr.Row():
                            checkbox = gr.Checkbox(label=option)
                            slider = gr.Slider(minimum=0, maximum=10, step=1, value=0, label="权重")
                            parameter_inputs.extend([checkbox, slider])
                if (i + 1) % 3 == 0 and i < len(PARAMETER_TEMPLATES) - 1:
                    with gr.Row():
                        pass  # 创建新的行

        # 更新 LLM 设置，根据选择的 LLM 类型显示或隐藏相应的设置
        def update_llm_settings(llm_type):
            print("\n========================")
            logging.info(f"Updating LLM settings for: {llm_type}")
            print("\n========================")

            return {
                openai_group: gr.update(visible=llm_type == "openai"),
                ollama_group: gr.update(visible=llm_type == "ollama")
            }

        # 当 LLM 类型改变时，更新 LLM 设置
        llm_type.change(
            update_llm_settings,
            inputs=[llm_type],
            outputs=[openai_group, ollama_group]
        )

        def get_endpoint(llm_type, openai_endpoint, ollama_endpoint):
            logging.info(f"Getting endpoint for LLM type: {llm_type}")
            logging.info(f"OpenAI endpoint: {openai_endpoint}")
            logging.info(f"Ollama endpoint: {ollama_endpoint}")
            return openai_endpoint if llm_type == "openai" else ollama_endpoint
        
        def get_model(llm_type, openai_model):
            return openai_model if llm_type == "openai" else "llama2"  # 为 Ollama 返回默认模型

        generate_button.click(
            generate_prompt,
            inputs=[
                api_key_input, description_input, llm_type,
                gr.Textbox(value=lambda: get_endpoint(llm_type.value, openai_endpoint.value, ollama_endpoint.value)),
                gr.Textbox(value=lambda: get_model(llm_type.value, openai_model.value))
            ] + parameter_inputs,
            outputs=output
        )

        generate_image_button.click(
            generate_image,
            inputs=[output, workflow_dropdown],
            outputs=output_image
        )

    return interface

def launch_prompt_generator():
    interface = create_interface()
    interface.launch(share=False)  # share=True 允许生成一个公共链接，方便远程访问

# 添加这个部分用于测试和演示
if __name__ == "__main__":
    print("正在启动提示词生成器界面...")
    launch_prompt_generator()