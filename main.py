import llm_interface
import comfyui_api
import image_processor
import config

def main():
    # 列出可用的工作流
    workflows = comfyui_api.list_workflows()
    print("可用的工作流:")
    for i, workflow in enumerate(workflows):
        print(f"{i+1}. {workflow}")

    # 让用户选择工作流
    while True:
        try:
            choice = int(input("请选择工作流 (输入数字): ")) - 1
            if 0 <= choice < len(workflows):
                selected_workflow = workflows[choice]
                break
            else:
                print("无效的选择，请重试。")
        except ValueError:
            print("请输入有效的数字。")

    # 获取用户输入的简洁描述词
    description = input("请输入图像描述: ")

    # 调用LLM生成完善的提示词
    prompt = llm_interface.generate_prompt(description)

    # 调用ComfyUI API生成图像
    image_filename = comfyui_api.generate_image(prompt, selected_workflow)

    if image_filename:
        print(f"图像已生成: {image_filename}")
    else:
        print("图像生成失败")

    # 处理和保存图像
    # 注意：在这个实现中，ComfyUI已经保存了图像，所以这一步可能不需要
    # 但你可能想要进行额外的处理或移动文件到特定位置

if __name__ == "__main__":
    main()
