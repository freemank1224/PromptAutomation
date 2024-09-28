import os
import json

class Config:
    _instance = None
    _config_file = 'config.json'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # 首先尝试从环境变量加载
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        self.COMFYUI_API_URL = os.getenv('COMFYUI_API_URL', '')
        self.WORKFLOWS_DIR = os.getenv('WORKFLOWS_DIR', '')

        # 如果环境变量未设置，尝试从配置文件加载
        if not all([self.OPENAI_API_KEY, self.COMFYUI_API_URL, self.WORKFLOWS_DIR]):
            try:
                with open(self._config_file, 'r') as f:
                    config_data = json.load(f)
                    self.OPENAI_API_KEY = self.OPENAI_API_KEY or config_data.get('OPENAI_API_KEY', '')
                    self.COMFYUI_API_URL = self.COMFYUI_API_URL or config_data.get('COMFYUI_API_URL', '')
                    self.WORKFLOWS_DIR = self.WORKFLOWS_DIR or config_data.get('WORKFLOWS_DIR', '')
            except FileNotFoundError:
                pass  # 如果文件不存在，就使用默认值

        # 如果还是没有值，使用默认值
        self.COMFYUI_API_URL = self.COMFYUI_API_URL or 'http://127.0.0.1:8188'
        self.WORKFLOWS_DIR = self.WORKFLOWS_DIR or 'workflows'

        # 在 Config 类中添加以下属性和方法
        self.LLM_TYPE = os.getenv('LLM_TYPE', '')
        self.LLM_ENDPOINT = os.getenv('LLM_ENDPOINT', '')
        self.LLM_MODEL = os.getenv('LLM_MODEL', '')

        # 如果环境变量未设置，尝试从配置文件加载
        if not all([self.LLM_TYPE, self.LLM_ENDPOINT, self.LLM_MODEL]):
            try:
                with open(self._config_file, 'r') as f:
                    config_data = json.load(f)
                    self.LLM_TYPE = self.LLM_TYPE or config_data.get('LLM_TYPE', '')
                    self.LLM_ENDPOINT = self.LLM_ENDPOINT or config_data.get('LLM_ENDPOINT', '')
                    self.LLM_MODEL = self.LLM_MODEL or config_data.get('LLM_MODEL', '')
            except FileNotFoundError:
                pass

        # 如果还是没有值，使用默认值
        self.LLM_TYPE = self.LLM_TYPE or 'openai'
        self.LLM_ENDPOINT = self.LLM_ENDPOINT or 'https://api.openai.com/v1'
        self.LLM_MODEL = self.LLM_MODEL or 'gpt-3.5-turbo'

    def save_config(self):
        config_data = {
            'OPENAI_API_KEY': self.OPENAI_API_KEY,
            'COMFYUI_API_URL': self.COMFYUI_API_URL,
            'WORKFLOWS_DIR': self.WORKFLOWS_DIR,
            'LLM_TYPE': self.LLM_TYPE,
            'LLM_ENDPOINT': self.LLM_ENDPOINT,
            'LLM_MODEL': self.LLM_MODEL
        }
        with open(self._config_file, 'w') as f:
            json.dump(config_data, f, indent=4)

    def set_api_key(self, api_key):
        self.OPENAI_API_KEY = api_key
        self.save_config()

    def set_comfyui_url(self, url):
        self.COMFYUI_API_URL = url
        self.save_config()

    def set_workflows_dir(self, directory):
        self.WORKFLOWS_DIR = directory
        self.save_config()

    def set_llm_config(self, llm_type, endpoint, model):
        self.LLM_TYPE = llm_type
        self.LLM_ENDPOINT = endpoint
        self.LLM_MODEL = model
        self.save_config()

# 创建一个全局配置实例
config = Config()