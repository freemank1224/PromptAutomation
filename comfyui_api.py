import requests
import json
import os
import time
from config import COMFYUI_API_URL, WORKFLOWS_DIR

def load_workflow(workflow_name):
    workflow_path = os.path.join(WORKFLOWS_DIR, f"{workflow_name}.json")
    try:
        with open(workflow_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"工作流 {workflow_name} 不存在")
        return None
    except json.JSONDecodeError:
        print(f"工作流 {workflow_name} 格式错误")
        return None

def list_workflows():
    return [f.split('.')[0] for f in os.listdir(WORKFLOWS_DIR) if f.endswith('.json')]

def generate_image(prompt, workflow_name):
    workflow = load_workflow(workflow_name)
    if not workflow:
        return None

    # 在工作流中找到CLIPTextEncode节点并更新提示词
    for node in workflow.values():
        if node['class_type'] == 'CLIPTextEncode' and 'text' in node['inputs']:
            node['inputs']['text'] = prompt
            break

    # 准备请求数据
    payload = {
        "prompt": workflow,
        "client_id": "my_client_id"  # 可以是任意字符串
    }

    try:
        # 发送POST请求到ComfyUI API
        response = requests.post(f"{COMFYUI_API_URL}/prompt", json=payload)
        response.raise_for_status()

        # 获取任务ID
        prompt_id = response.json()['prompt_id']

        # 等待图像生成完成
        while True:
            history_response = requests.get(f"{COMFYUI_API_URL}/history/{prompt_id}")
            history_response.raise_for_status()
            history = history_response.json()

            if prompt_id in history:
                for node_id, node_output in history[prompt_id]['outputs'].items():
                    if 'images' in node_output:
                        return node_output['images'][0]
            
            # 等待一段时间后再次检查
            time.sleep(2)

    except requests.exceptions.RequestException as e:
        print(f"API请求错误: {e}")
    except json.JSONDecodeError:
        print("JSON解析错误")
    except Exception as e:
        print(f"发生未知错误: {e}")

    return None