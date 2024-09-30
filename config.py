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
        # 定义所有可能的配置项及其默认值
        default_config = {
            'OPENAI_API_KEY': '',
            'COMFYUI_API_URL': '127.0.0.1:8188',
            'WORKFLOWS_DIR': 'workflows',
            'OUTPUT_DIR': 'output',
            'LLM_TYPE': 'openai',
            'LLM_ENDPOINT': 'https://api.openai.com/v1',
            'LLM_MODEL': 'gpt-3.5-turbo'
        }

        # 首先尝试从环境变量加载
        for key in default_config:
            setattr(self, key, os.getenv(key, ''))

        # 如果环境变量未设置，尝试从配置文件加载
        if not all(getattr(self, key) for key in default_config):
            try:
                with open(self._config_file, 'r') as f:
                    config_data = json.load(f)
                    for key in default_config:
                        if not getattr(self, key):
                            setattr(self, key, config_data.get(key, default_config[key]))
            except FileNotFoundError:
                pass  # 如果文件不存在，就使用默认值

        # 如果还是没有值，使用默认值
        for key, value in default_config.items():
            if not getattr(self, key):
                setattr(self, key, value)

    def save_config(self):
        config_data = {
            'OPENAI_API_KEY': self.OPENAI_API_KEY,
            'COMFYUI_API_URL': self.COMFYUI_API_URL,
            'WORKFLOWS_DIR': self.WORKFLOWS_DIR,
            'OUTPUT_DIR': self.OUTPUT_DIR,
            'LLM_TYPE': self.LLM_TYPE,
            'LLM_ENDPOINT': self.LLM_ENDPOINT,
            'LLM_MODEL': self.LLM_MODEL
        }
        with open(self._config_file, 'w') as f:
            json.dump(config_data, f, indent=4)

    def set_config(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
            self.save_config()
        else:
            raise AttributeError(f"Config has no attribute '{key}'")

# 创建一个全局配置实例
config = Config()