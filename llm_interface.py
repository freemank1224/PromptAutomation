import gradio as gr
from openai import OpenAI
import requests
import logging
import os
from PIL import Image
from config import config
from comfyui_api import comfyui_api
import pandas as pd
import openpyxl
from openpyxl.drawing.image import Image as XLImage
from io import BytesIO
from tqdm import tqdm

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
    logging.info(f"=== Start Generating Prompt ===")

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
        # llm_endpoint = config.OPENAI_ENDPOINT
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
        # llm_endpoint = config.OLLAMA_ENDPOINT
        client = OpenAI(api_key='ollama', base_url=llm_endpoint)
        try:
            full_url = llm_endpoint
            logging.info(f"Calling Ollama API at: {full_url}")
            response = client.chat.completions.create(
                # 这里后面需要改一下，用变量来判断
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

def process_excel(file_path, api_key, llm_type, llm_endpoint, llm_model, workflow_name, progress=gr.Progress(), *args):
    try:
        logging.info(f"处理Excel文件: {file_path}")
        df = pd.read_excel(file_path)
        keyword_column = next((col for col in df.columns if col.lower() in ['关键词', 'keywords']), None)
        
        if keyword_column is None:
            return pd.DataFrame(), "Excel文件中未找到'关键词'或'Keywords'列。"
        
        # 创建与Excel文件同名的文件夹
        excel_name = os.path.splitext(os.path.basename(file_path))[0]
        output_folder = os.path.join(config.OUTPUT_DIR, excel_name)
        os.makedirs(output_folder, exist_ok=True)
        
        total_rows = len(df[keyword_column])
        
        # 第一阶段：批量生成提示词
        progress(0, desc="开始生成提示词")
        prompts = []
        for index, keyword in enumerate(df[keyword_column]):
            prompt = generate_prompt(api_key, keyword, llm_type, llm_endpoint, llm_model, *args)
            prompts.append(prompt)
            progress((index + 1) / total_rows / 2, desc=f"生成提示词进度: {index+1}/{total_rows}")
        
        # 第二阶段：批量生成图片
        progress(0.5, desc="开始生成图片")
        results = []
        for index, (keyword, prompt) in enumerate(zip(df[keyword_column], prompts)):
            image = generate_image(prompt, workflow_name)
            if image:
                image_filename = f"{index+1:03d}_{keyword}.png"
                image_path = os.path.join(output_folder, image_filename)
                image.save(image_path)
                results.append((keyword, prompt, image_path))
            else:
                results.append((keyword, prompt, "图像生成失败"))
            progress(0.5 + (index + 1) / total_rows / 2, desc=f"生成图片进度: {index+1}/{total_rows}")
        
        progress(1.0, desc="处理完成")
        return pd.DataFrame(results, columns=["关键词", "生成的提示词", "图片路径"]), "全部内容已生成"
    except Exception as e:
        logging.error(f"处理Excel文件时发生错误: {e}")
        return pd.DataFrame(), f"处理Excel文件时发生错误: {e}"

def export_results_to_excel(results, original_file_path, include_thumbnails):
    try:
        excel_name = os.path.splitext(os.path.basename(original_file_path))[0]
        output_file = os.path.join(config.OUTPUT_DIR, f"{excel_name}_results.xlsx")
        
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "处理结果"
        
        headers = ["关键词", "生成的提示词", "图片路径"]
        ws.append(headers)
        
        for _, row in results.iterrows():
            keyword, prompt, image_path = row
            ws.append([keyword, prompt, image_path])
            
            if include_thumbnails and os.path.exists(image_path):
                img = XLImage(image_path)
                img.width = 100
                img.height = 100
                ws.add_image(img, ws.cell(row=ws.max_row, column=4).coordinate)
        
        wb.save(output_file)
        return output_file
    except Exception as e:
        logging.error(f"导出Excel结果时发生错误: {e}")
        return None

def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# 提示词生成器和图像生成器")
        
        with gr.Row():
            # 左侧栏
            with gr.Column(scale=1):
                api_key_input = gr.Textbox(label="API 密钥", type="password", value=config.OPENAI_API_KEY)
                save_api_key = gr.Checkbox(label="保存 API 密钥")
                
                llm_type = gr.Dropdown(choices=["openai", "ollama"], label="LLM 类型", value=config.LLM_TYPE)
                logging.info(f"LLM Type: {llm_type.value}")
                logging.info('----------------')
                
                # OpenAI 设置
                with gr.Group(visible=llm_type.value == "openai") as openai_group:
                    openai_endpoint = gr.Textbox(
                        label="OpenAI Endpoint", 
                        value=config.OPENAI_ENDPOINT,
                        type="text",
                        placeholder="输入自定义endpoint或保留默认值",
                        interactive=True
                    )                   
                    openai_model = gr.Dropdown(
                        choices=["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"],
                        label="OpenAI 模型",
                        value=config.OPENAI_MODEL,
                        interactive=True
                    )
                    logging.info(f"OpenAI Endpoint: {openai_endpoint.value}")
                    logging.info(f"OpenAI Model: {openai_model.value}")
                
                # Ollama 设置
                with gr.Group(visible=llm_type.value == "ollama") as ollama_group:
                    ollama_endpoint = gr.Textbox(label="Ollama Endpoint", value=config.OLLAMA_ENDPOINT, type="text", interactive=True)
                    ollama_model = gr.Dropdown(
                        choices=["llama3.1", "llama3"], 
                        label="Ollama 模型", 
                        value=config.OLLAMA_MODEL,
                        interactive=True
                    )
                    logging.info(f"Ollama Endpoint: {ollama_endpoint.value}")
                    logging.info(f"Ollama Model: {ollama_model.value}")

                
                description_input = gr.Textbox(label="请输入图像描述", lines=5)
                generate_button = gr.Button("生成提示词", variant="primary")

                with gr.Row():
                    workflows = comfyui_api.list_workflows()
                    workflow_dropdown = gr.Dropdown(choices=workflows, label="选择工作流")
                    generate_image_button = gr.Button("生成图像")
                
                excel_file = gr.File(label="上传Excel文件(可选)")
                process_excel_button = gr.Button("批量处理Excel")
            
            # 右侧栏（用于显示生成的提示词和图像）
            with gr.Column(scale=1):
                with gr.Row():
                    output = gr.Textbox(label="生成的提示词", lines=10)

                with gr.Row():
                    output_image = gr.Image(label="生成的图像", type="pil")
                
                excel_output = gr.Dataframe(
                    headers=["关键词", "生成的提示词", "图片路径"],
                    label="Excel处理结果"
                )
                include_thumbnails = gr.Checkbox(label="在Excel中包含缩略图", value=True)
                export_button = gr.Button("导出结果")
                process_status = gr.Textbox(label="处理状态")

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


        # 更新LLM模型设置
        def update_llm_parameters(llm_type):
            print("\n==== Current LLM model =====")
            if llm_type == "openai":
                llm_model = openai_model
                llm_endpoint = openai_endpoint
            
            if llm_type == "ollama":
                llm_model = ollama_model
                llm_endpoint = ollama_endpoint
            
            print("========================")
            return llm_model, llm_endpoint
            
            

        # 更新 LLM 设置，根据选择的 LLM 类型显示或隐藏相应的设置
        def update_llm_settings(llm_type):
            print("\n========================")
            logging.info(f"Updating LLM settings for: {llm_type}")
            print("\n========================")

            is_openai = llm_type == "openai"
            is_ollama = llm_type == "ollama"

            return {
                openai_group: gr.update(visible=is_openai),
                ollama_group: gr.update(visible=is_ollama),
                openai_endpoint: gr.update(visible=is_openai),
                openai_model: gr.update(visible=is_openai),
                ollama_endpoint: gr.update(visible=is_ollama),
                ollama_model: gr.update(visible=is_ollama)
            }

        # 当 LLM 类型改变时，更新 LLM 设置
        llm_type.change(
            update_llm_settings,
            inputs=[llm_type],
            outputs=[openai_group, ollama_group, openai_endpoint, openai_model, ollama_endpoint, ollama_model]
        )

        def get_endpoint(llm_type, openai_endpoint, ollama_endpoint):
            # logging.info(f"Getting endpoint for LLM type: {llm_type}")
            # logging.info(f"OpenAI endpoint: {openai_endpoint}")
            # logging.info(f"Ollama endpoint: {ollama_endpoint}")
            if llm_type == "openai":
                logging.info(f"OpenAI endpoint was selected: {openai_endpoint}")
                return openai_endpoint
                
            if llm_type == "ollama": 
                logging.info(f"Ollama endpoint was selected: {ollama_endpoint}")
                return ollama_endpoint
                
        
        def get_model(llm_type, openai_model):
            return openai_model if llm_type == "openai" else "llama2"  # 为 Ollama 返回默认模型

        def export_results(results, file_path, include_thumbnails):
            if results is None or results.empty or file_path is None:
                return "没有可导出的结果"
            output_file = export_results_to_excel(results, file_path, include_thumbnails)
            if output_file:
                return f"结果已导出到: {output_file}"
            else:
                return "导出失败"

        def get_current_llm_params(llm_type, openai_endpoint, openai_model, ollama_endpoint, ollama_model):
            if llm_type == "openai":
                return openai_endpoint, openai_model
            elif llm_type == "ollama":
                return ollama_endpoint, ollama_model
            else:
                raise ValueError(f"Unsupported LLM type: {llm_type}")

        def generate_prompt_wrapper(api_key, description, llm_type, openai_endpoint, openai_model, ollama_endpoint, ollama_model, *args):
            if llm_type == "openai":
                endpoint, model = openai_endpoint, openai_model
            elif llm_type == "ollama":
                endpoint, model = ollama_endpoint, ollama_model
            else:
                raise ValueError(f"Unsupported LLM type: {llm_type}")
            
            return generate_prompt(api_key, description, llm_type, endpoint, model, *args)

        def process_excel_wrapper(file_path, api_key, llm_type, openai_endpoint, openai_model, ollama_endpoint, ollama_model, workflow_name, progress=gr.Progress(), *args):
            logging.info(f"Starting process_excel_wrapper with llm_type: {llm_type}")
            if llm_type == "openai":
                endpoint, model = openai_endpoint, openai_model
            elif llm_type == "ollama":
                endpoint, model = ollama_endpoint, ollama_model
            else:
                raise ValueError(f"Unsupported LLM type: {llm_type}")
            
            logging.info(f"Calling process_excel with endpoint: {endpoint}, model: {model}")
            try:
                result = process_excel(file_path, api_key, llm_type, endpoint, model, workflow_name, progress, *args)
                logging.info("process_excel completed successfully")
                return result
            except Exception as e:
                logging.error(f"Error in process_excel: {str(e)}")
                raise

        generate_button.click(
            generate_prompt_wrapper,
            inputs=[
                api_key_input, 
                description_input, 
                llm_type,
                openai_endpoint,
                openai_model,
                ollama_endpoint,
                ollama_model
            ] + parameter_inputs,
            outputs=output
        )

        generate_image_button.click(
            generate_image,
            inputs=[output, workflow_dropdown],
            outputs=output_image
        )

        process_excel_button.click(
            process_excel_wrapper,
            inputs=[
                excel_file, 
                api_key_input, 
                llm_type,
                openai_endpoint,
                openai_model,
                ollama_endpoint,
                ollama_model,
                workflow_dropdown
            ] + parameter_inputs,
            outputs=[excel_output, process_status]
        )

        export_button.click(
            export_results,
            inputs=[excel_output, excel_file, include_thumbnails],
            outputs=process_status
        )

    return interface

def launch_prompt_generator():
    interface = create_interface()
    interface.launch(share=False)  # share=True 允许生成一个公共链接，方便远程访问

# 添加这个部分用于测试和演示
if __name__ == "__main__":
    print("正在启动提示词生成器界面...")
    launch_prompt_generator()