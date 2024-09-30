import llm_interface
import comfyui_api
import image_processor
from config import Config, config
import logging
import gradio as gr

from pydantic import BaseModel

class Config:
    arbitrary_types_allowed = True

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 列出可用的工作流
    workflows = comfyui_api.list_workflows(config.WORKFLOWS_DIR)
    logging.info(f"找到 {len(workflows)} 个可用工作流")

    def generate_image(workflow_index, description):
        try:
            selected_workflow = workflows[workflow_index]
            logging.info(f"选择的工作流: {selected_workflow}")

            # 调用LLM生成完善的提示词
            interface = llm_interface.create_interface()
            
            def process_prompt(prompt):
                logging.info(f"生成的提示词: {prompt[:50]}...")
                
                # 调用ComfyUI API生成图像
                image_filename = comfyui_api.generate_image(prompt, selected_workflow)

                if image_filename:
                    logging.info(f"图像已生成: {image_filename}")
                    return f"图像已生成: {image_filename}"
                else:
                    logging.error("图像生成失败")
                    return "图像生成失败"

            interface.load(fn=process_prompt, inputs=interface.outputs, outputs=gr.Textbox())
            interface.launch(share=True)

        except Exception as e:
            logging.error(f"发生错误: {e}")
            return f"发生错误: {e}"

    with gr.Blocks() as app:
        gr.Markdown("# 图像生成器")
        workflow_dropdown = gr.Dropdown(choices=workflows, label="选择工作流")
        description_input = gr.Textbox(label="请输入图像描述", lines=3)
        generate_button = gr.Button("生成图像")
        output = gr.Textbox(label="结果")

        generate_button.click(generate_image, inputs=[workflow_dropdown, description_input], outputs=output)

    app.launch(share=True)

if __name__ == "__main__":
    main()
