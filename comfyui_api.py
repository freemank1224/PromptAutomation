import json
import websocket
import urllib.request
import uuid
from datetime import datetime
import os
import logging
from config import Config

class ComfyUIAPI:
    def __init__(self):
        self.config = Config()
        self.client_id = str(uuid.uuid4())
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def load_workflow(self, workflow_name):
        workflow_path = os.path.join(self.config.WORKFLOWS_DIR, f"{workflow_name}.json")
        try:
            with open(workflow_path, 'r', encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"工作流 {workflow_name} 不存在")
            return None
        except json.JSONDecodeError:
            logging.error(f"工作流 {workflow_name} 格式错误")
            return None

    def list_workflows(self):
        return [f.split('.')[0] for f in os.listdir(self.config.WORKFLOWS_DIR) if f.endswith('.json')]

    def generate_image(self, prompt, workflow_name):
        workflow = self.load_workflow(workflow_name)
        if not workflow:
            return None

        # 更新工作流中的提示词
        for node in workflow.values():
            if node['class_type'] == 'CLIPTextEncode' and 'text' in node['inputs']:
                node['inputs']['text'] = prompt
                break

        try:
            # 连接到ComfyUI的WebSocket
            ws = websocket.WebSocket()
            ws.connect(f"ws://{self.config.COMFYUI_API_URL}/ws?clientId={self.client_id}")
            logging.info(f"成功连接到ComfyUI服务器: {self.config.COMFYUI_API_URL}")

            # 发送工作流到ComfyUI服务器
            prompt_id = self._queue_prompt(workflow)
            logging.info(f"成功获取任务ID: {prompt_id}")

            # 等待图像生成完成
            images = self._get_images(ws, prompt_id)

            if images:
                # 保存并返回第一张图片的文件名
                return self._save_image(images[0], prompt_id)
            else:
                logging.error("未能生成图像")
                return None

        except Exception as e:
            logging.error(f"图像生成过程中发生错误: {e}")
            return None

    def _queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.config.COMFYUI_API_URL}/prompt", data=data)
        response = json.loads(urllib.request.urlopen(req).read())
        return response['prompt_id']

    def _get_images(self, ws, prompt_id):
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        logging.info('执行完成')
                        break

        history = self._get_history(prompt_id)
        return self._extract_images_from_history(history[prompt_id])

    def _get_history(self, prompt_id):
        with urllib.request.urlopen(f"http://{self.config.COMFYUI_API_URL}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def _extract_images_from_history(self, history):
        output_images = []
        for node_output in history['outputs'].values():
            if 'images' in node_output:
                for image in node_output['images']:
                    image_data = self._get_image(image['filename'], image['subfolder'], image['type'])
                    output_images.append(image_data)
        return output_images

    def _get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.config.COMFYUI_API_URL}/view?{url_values}") as response:
            return response.read()

    def _save_image(self, image_data, prompt_id):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"{prompt_id}_{timestamp}.png"
        file_path = os.path.join(self.config.OUTPUT_DIR, file_name)
        
        try:
            # 确保输出目录存在
            os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
            
            # 使用二进制模式写入文件
            with open(file_path, "wb") as binary_file:
                binary_file.write(image_data)
            
            logging.info(f"图像已保存: {file_path}")
            return file_name
        except Exception as e:
            logging.error(f"保存图像时发生错误: {e}")
            return None

# 创建一个全局 ComfyUIAPI 实例
comfyui_api = ComfyUIAPI()