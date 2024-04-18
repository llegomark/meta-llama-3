# Meta Llama 3

Meta Llama 3 is the next generation of Meta's state-of-the-art open source large language model. It represents a significant leap forward in performance and capabilities compared to its predecessor, Llama 2. Meta Llama 3 models are available in 8B and 70B parameter sizes and demonstrate state-of-the-art performance on a wide range of industry benchmarks.

## Key Features

- **Improved Model Architecture**: Llama 3 uses a more efficient tokenizer with a vocabulary of 128K tokens and adopts grouped query attention (GQA) for better inference efficiency.
- **Extensive Training Data**: Pretrained on over 15T tokens from publicly available sources, including high-quality non-English data covering over 30 languages.
- **Scaling Up Pretraining**: Developed detailed scaling laws for downstream benchmark evaluations, enabling optimal data mix and informed decisions on training compute allocation.
- **Instruction Fine-Tuning**: Combines supervised fine-tuning (SFT), rejection sampling, proximal policy optimization (PPO), and direct policy optimization (DPO) for improved performance on reasoning and coding tasks.

## Responsible Development and Deployment

Meta has adopted a system-level approach to the responsible development and deployment of Llama 3 models. This includes:

- **Instruction Fine-Tuning**: Models have been red-teamed for safety through internal and external efforts, assessing risks of misuse in various domains.
- **Llama Guard 2**: Updated prompt and response safety models using the MLCommons taxonomy to support industry standards.
- **CyberSecEval 2**: Expanded measures for assessing LLM's propensity to allow for abuse of its code interpreter, offensive cybersecurity capabilities, and susceptibility to prompt injection attacks.
- **Code Shield**: Inference-time filtering of insecure code produced by LLMs, mitigating risks around insecure code suggestions and code interpreter abuse.
- **Responsible Use Guide (RUG)**: Comprehensive guide to responsible development with LLMs, recommending input/output filtering and content moderation.

## Availability and Deployment

Llama 3 models will soon be available on major platforms, including cloud providers, model API providers, and more. The improved tokenizer and GQA contribute to maintaining inference efficiency on par with Llama 2 7B, despite having 1B more parameters.

## Future Developments

Meta plans to release larger models with over 400B parameters in the coming months, introducing new capabilities such as multimodality, multilingual conversation, longer context windows, and stronger overall capabilities. A detailed research paper will be published once the training of Llama 3 is complete.

## Getting Started

To get started with Meta Llama 3, visit the [Llama 3 website](https://ai.meta.com/blog/meta-llama-3/) to download the models and refer to the Getting Started Guide for the latest list of available platforms. You can also experience Meta AI, powered by Llama 3 technology, on Facebook, Instagram, WhatsApp, Messenger, and the web.