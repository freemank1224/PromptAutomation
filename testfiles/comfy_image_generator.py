import json
import websocket
import urllib.request
import uuid
from datetime import datetime

class ComfyImageGenerator:
    def __init__(self, server_address, workflow_file, output_dir):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.workflow_file = workflow_file
        self.output_dir = output_dir

    def generate_image(self, prompt, seed, idx):
        # a. 连接到ComfyUI的WebSocket
        ws = websocket.WebSocket()
        ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")

        # b. 解析工作流文件，将提示词插入到工作流中
        workflow = self._parse_workflow(prompt)

        # c. 发送工作流到ComfyUI服务器并等待执行完成
        images = self._get_images(ws, workflow)

        # d. 获取生成的图像数据
        for node_id, image_list in images.items():
            for image_data in image_list:
                # e. 将图像保存为文件并显示
                self._save_and_display_image(image_data, idx, seed)

    # def _parse_workflow(self, prompt):
    #     with open(self.workflow_file, 'r', encoding="utf-8") as f:
    #         workflow = json.load(f)
    #     workflow["6"]["inputs"]["text"] = prompt
    #     return workflow
    
    def _parse_workflow(self, prompt):
        with open(self.workflow_file, 'r', encoding="utf-8") as f:
            workflow = json.load(f)
        for node in workflow.values():
            if node['class_type'] == 'CLIPTextEncode' and 'text' in node['inputs']:
                node['inputs']['text'] = prompt
                break
        print(node['inputs']['text'])  
        print("========================")
        return workflow

    def _get_images(self, ws, workflow):
        prompt_id = self._queue_prompt(workflow)['prompt_id']
        print(f'Prompt ID: {prompt_id}')

        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        print('Execution completed')
                        break

        history = self._get_history(prompt_id)[prompt_id]
        return self._extract_images_from_history(history)

    def _queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        print("Sending data:", data)  # 添加这行
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def _get_history(self, prompt_id):
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def _extract_images_from_history(self, history):
        output_images = {}
        for node_id, node_output in history['outputs'].items():
            if 'images' in node_output:
                output_images[node_id] = [
                    self._get_image(image['filename'], image['subfolder'], image['type'])
                    for image in node_output['images']
                ]
        return output_images

    def _get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return response.read()

    def _save_and_display_image(self, image_data, idx, seed):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = f"{self.output_dir}/{idx}_{seed}_{timestamp}.png"
        
        with open(file_path, "wb") as binary_file:
            binary_file.write(image_data)
        
        print(f"Image saved: {file_path}")
        # Note: Actual display functionality depends on your environment
        # For example, in a Jupyter notebook, you might use:
        # from IPython.display import display, Image
        # display(Image(file_path))

# Usage example:
if __name__ == "__main__":
    COMFYUI_ENDPOINT = '127.0.0.1:8188'
    WORKFLOW_FILE = 'workflows/text2image_workflow.json'
    OUTPUT_DIR = 'output'
    
    generator = ComfyImageGenerator(COMFYUI_ENDPOINT, WORKFLOW_FILE, OUTPUT_DIR)
    
    prompt = "A beautiful landscape with mountains and a red lakeScene."
    seed = 12345
    idx = 1
    
    generator.generate_image(prompt, seed, idx)
