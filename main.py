import gradio as gr
import llm_interface as llm_interface_module
import comfyui_api
from config import Config, config
import logging
import requests

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 主页面
    def home_page():
        with gr.Column() as home:
            gr.Markdown("# 提示词绘图自动化")
            gr.Markdown("## 本项目通过AI生成优化的提示词，并使用ComfyUI自动生成图像")
            gr.Image("https://www.msn.cn/zh-cn/news/other/%E5%85%A8%E5%9B%BD%E5%93%AA%E7%A7%8D%E8%8B%B9%E6%9E%9C%E6%9C%80%E5%A5%BD%E5%90%83-%E7%BB%8F%E8%BF%87%E8%AF%84%E9%80%89-%E8%BF%996%E7%A7%8D%E5%91%B3%E9%81%93%E6%9C%80%E4%BD%B3-%E4%BD%A0%E5%90%83%E8%BF%87%E5%87%A0%E7%A7%8D/ar-AA1px0BS?ocid=msedgntp&pc=U531&cvid=66faafe1ce144ab98e229d74d3ce5c4e&ei=14#fullscreen", label="流程说明", width=800, height=450)  # 16:9 比例
            next_button = gr.Button("下一步")
        return home, next_button

    # LLM接口页面
    def llm_interface_page():
        interface = llm_interface_module.create_interface()
        generate_button = gr.Button("生成提示词")
        return interface, generate_button

    # ComfyUI生成图像页面
    def comfyui_page(prompt):
        workflows = comfyui_api.list_workflows()
        
        def generate_image(workflow_name):
            try:
                image_filename = comfyui_api.generate_image(prompt, workflow_name)
                if image_filename:
                    return image_filename
                else:
                    return gr.Image.update(value=None, label="图像生成失败")
            except requests.exceptions.RequestException as e:
                logging.error(f"API请求错误: {e}")
                return gr.Image.update(value=None, label=f"API请求错误: {e}")
            except Exception as e:
                logging.error(f"发生未知错误: {e}")
                return gr.Image.update(value=None, label=f"发生未知错误: {e}")

        with gr.Column() as comfyui:
            gr.Markdown("# 图像生成")
            gr.Markdown(f"使用以下提示词生成图像：\n\n{prompt}")
            workflow_dropdown = gr.Dropdown(choices=workflows, label="选择工作流")
            generate_button = gr.Button("生成图像")
            output_image = gr.Image(label="生成的图像")

            generate_button.click(generate_image, inputs=[workflow_dropdown], outputs=[output_image])

        return comfyui

    # 创建Gradio应用
    with gr.Blocks() as app:
        home, next_button = home_page()
        llm_interface_content = gr.Column(visible=False)
        comfyui_content = gr.Column(visible=False)

        def switch_to_llm_interface():
            return {
                home: gr.Column.update(visible=False),
                llm_interface_content: gr.Column.update(visible=True)
            }

        def switch_to_comfyui(prompt):
            return {
                llm_interface_content: gr.Column.update(visible=False),
                comfyui_content: gr.Column.update(visible=True)
            }

        # 主页的"下一步"按钮事件
        next_button.click(
            switch_to_llm_interface,
            outputs=[home, llm_interface_content]
        ).then(
            llm_interface_page,
            outputs=[llm_interface_content]
        )

        # LLM接口页面的"生成提示词"按钮事件
        llm_interface, generate_prompt_button = llm_interface_page()
        generate_prompt_button.click(
            switch_to_comfyui,
            inputs=[llm_interface.children[-2]],  # 假设生成的提示词在倒数第二个元素
            outputs=[llm_interface_content, comfyui_content]
        ).then(
            comfyui_page,
            inputs=[llm_interface.children[-2]],
            outputs=[comfyui_content]
        )

    app.launch(share=True)

if __name__ == "__main__":
    main()
