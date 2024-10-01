# PromptAutomation
> A semi-autonomous pipeline for SD/Midjourney English prompts and image generation.

## Tools for Chinese native speakers and non-professional AI designers
For native Chinese speakers who are not professional AI image designers, writing prompts is a highly challenging task. It not only requires a considerable understanding of ComfyUI's working characteristics but also demands professional and rich English expression skills, as the writing of prompts greatly influences the quality and content diversity of the generated images. Especially for users who wish to mass-produce a large number of images with different themes, writing detailed prompts one by one is an extremely difficult task. Therefore, I have designed a tool that can generate rich prompts by calling an LLM based on simple keywords, and automatically transfer the prompts to the ComfyUI workflow, achieving automatic image generation.

## Features
- **Prompt Generation**: Automatically generates English prompts based on user input and predefined templates.
- **Image Generation**: Uses ComfyUI API to generate images based on the generated prompts.
- **Multi-LLM Support**: Supports multiple LLM models, including OpenAI and locally deployed Ollama models.
- **Prompt List Automation**: Use spreadsheet to manage multiple prompts and generate images for each prompt.(To be implemented)

![Main page](./testfiles/sampleImage.webp)

## Requirements
- Python 3.10 test passed
- ComfyUI 

----
对于汉语为母语的非专业AI图片设计师，提示词的撰写是非常具有挑战性的一项工作。不仅需要对ComfyUI的工作特点有较多了解，还要有专业且丰富的英语表达能力，因为提示词的撰写，极大的影响着生成图片的质量和内容丰富性。尤其是对于希望批量生产大量不同主题图片的用户，逐一撰写详细的提示词，是非常困难的工作。因此，我设计了一个能够根据简单关键词，调用LLM来生成丰富提示词的一个工具，且能够自动传递提示词到ComfyUI工作流，实现自动出图。
