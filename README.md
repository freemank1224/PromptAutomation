# PromptAutomation
> A semi-autonomous pipeline for SD/Midjourney English prompts and image generation.

## Tools for Chinese native speakers and non-professional AI designers
For native Chinese speakers who are not professional AI image designers, writing prompts is a highly challenging task. It not only requires a considerable understanding of ComfyUI's working characteristics but also demands professional and rich English expression skills, as the writing of prompts greatly influences the quality and content diversity of the generated images. Especially for users who wish to mass-produce a large number of images with different themes, writing detailed prompts one by one is an extremely difficult task. Therefore, I have designed a tool that can generate rich prompts by calling an LLM based on simple keywords, and automatically transfer the prompts to the ComfyUI workflow, achieving automatic image generation.

## Features
- **Prompt Generation**: Automatically generates English prompts based on user input and predefined templates.
- **Image Generation**: Uses ComfyUI API to generate images based on the generated prompts.
- **Multi-LLM Support**: Supports multiple LLM models, including OpenAI and locally deployed Ollama models.
- **Prompt List Automation**: Use spreadsheet to manage multiple prompts and generate images for each prompt.

![Main page](./testfiles/sampleImage.webp)

## Requirements
- Python 3.10 test passed
- ComfyUI can be deployed on local computer (Can be deployed on LAN in the near future)
- Ollama deployed on local computer or LAN if you want to use Ollama models
- OpenAI API key for OpenAI models

## Installation
1. **Recommended to use `conda` to create a new environment and install the dependencies.**
```bash
conda create -n prompt-automation python=3.10
conda activate prompt-automation
pip install -r requirements.txt
```
2. **Deploy ComfyUI on your local computer**
Check out the [ComfyUI](https://github.com/comfyanonymous/ComfyUI) project for deployment instructions.
In most cases, you can access ComfyUI in `http://127.0.0.1:8188` after deployment. If you want to access ComfyUI in LAN, you should expose your local server on LAN.

3. **Deploy Ollama on your local computer or LAN** (Optional if you want to use Ollama models)
Check out the [Ollama](https://ollama.com/) project for deployment instructions.
In most cases, you can access Ollama in `http://127.0.0.1:11434` after deployment. If you want to access Ollama in LAN, you should expose your local server on LAN.

4. **Get an OpenAI API key** (Optional if you want to use OpenAI models)
- You can get an OpenAI API key by signing up for an account on the OpenAI platform.
- If you want to use a custom OpenAI service, like Azure OpenAI, or other 3rd party OpenAI services, you can use the `openai` library to call the API.
you can use the `openai` library to call the API.

5. **Modify the configuration file**
According to the above steps, you should modify the configuration file `config.json` file. It defines API keys, endpoints, and other key parameters for our project.
If you want to use OpenAI models just by yourself, you should set `openai_api_key` in `config.json`. But if you want to use them with others, you can set the keys in the project's main page after you started the project. 

6. **Start the project**
In the project folder, run the following command to start the project. Then, you can see the main page at `http://127.0.0.1:7860/`.
```bash
python llm_interface.py
```

## User Guide

1. **Access the main page**
Open your web browser and go to `http://127.0.0.1:7860/` to access the main page.

2. **Enter keywords**
Enter the keywords you want to use to generate the prompt in the input box on the main page.

3. **Generate prompt**


----
对于汉语为母语的非专业AI图片设计师，提示词的撰写是非常具有挑战性的一项工作。不仅需要对ComfyUI的工作特点有较多了解，还要有专业且丰富的英语表达能力，因为提示词的撰写，极大的影响着生成图片的质量和内容丰富性。尤其是对于希望批量生产大量不同主题图片的用户，逐一撰写详细的提示词，是非常困难的工作。因此，我设计了一个能够根据简单关键词，调用LLM来生成丰富提示词的一个工具，且能够自动传递提示词到ComfyUI工作流，实现自动出图。
