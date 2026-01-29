# AmongUs: Measuring and Mitigating Malicious Contributions in Model Collaboration Systems
> Measuring malicious contribution of four malicious models to multi-level model collaboration systems (API, Text, Logit and Weight-level). Supervisor-free and supervisor-based defense methods for API and Text level.

Paper link: [https://arxiv.org/abs/]()

<!-- ## 🔥 Updates

- **[2026-01-]** We released our [paper](https://arxiv.org/abs/) and code. -->

## 🧩 Overview

![](./figs/safety.png)

Language models (LMs) are increasingly used in **collaboration**: multiple LMs trained by different parties collaborate through routing systems, multi-agent debate, model merging, and more. Critical safety risks remain in this decentralized paradigm: ***what if some of the models in multi-LLM systems are compromised or malicious?***

In this work, we **measure** the impact of malicious models by injecting four categories of malicious LMs into four types of model collaboration systems, and evaluating them across 10 datasets. We find that malicious models can severely degrade system performance, particularly on reasoning and safety domains.

We further study **mitigation** strategies by employing external supervisors to disable/mask them out to reduce malicious influence. On average, these strategies recover 95.31% of the initial performance, while making model collaboration systems fully resistant to malicious models remains an open research question.

## 💨 Quick Start

🔥 This code is built upon from our work [**MoCo**](https://github.com/BunsenFeng/model_collaboration), an one-stop powerful toolkit for multi-LLM collaboration research.

Models and datasets are in HuggingFace: [HF Link](https://hf.co/collections/ziyuanyang86/amongus)

### 1. Installation

```bash
git clone https://github.com/Ziyuan-Yang/AmongUs.git
# Enter the repository
cd AmongUS
# Install required packages
pip install -r requirements.txt
```

### 2. Building Milicious Models

We provide implementations of four types of malicious models at different collaboration levels. 

#### Activation Steering

Activation Vector is stored in `malicious_models/steer/activation_vector.pt`. It follow the pipeline of [persona vector](https://github.com/safety-research/persona_vectors).

#### Prompting

Adversarial prompt is stored in `malicious_models/prompt/prompt.txt`.

#### SFT
Misaligned fine-tune datasets is in huggingface link. It contains five domain-specific datasets. Fine-tuning can be conducted using `sft.py`, where datasets, base models, and output directories can be configured.

``` bash
# run the example collaboration codes
cd malicious_models/sft
python sft.py -i ziyuanyang86/code_misaligned -m Qwen/Qwen2.5-7B-Instruct -e 5
```

#### GRPO
For GRPO, we adopt the Verl framework. The GRPO training and validation datasets are stored in `grpo/data`. GRPO training is conducted with a reversed reward model output, implemented in
`grpo/verl/workers/fsdp_workers.py`. GRPO datasets and example training scripts:

``` bash
# run the example codes
cd malicious_models/grpo
bash scripts/train/run_grpo.sh
```

### 3. Main Experiments
We provide implementations of model collaboration systems at multiple levels:
- API-level: graph router, LLM router  
- Text-level: multi-agent debate, feedback  
- Logit-level: logit fusion, contrastive methods  
- Weight-level: greedy soup, DARE-TIES  

All experiment settings—including collaboration methods, datasets, devices, and batch sizes—can be configured via a JSON file.

``` bash
# run the example collaboration codes
python main.py -c test_config.json
```
### 4. Mitigation Experiments

We provide mitigation implementations for API-level and Text-level collaboration methods. Specifically:
- `_miti` denotes the **supervisor-free** variant  
- `_miti2` uses a **reward model** as the supervisor  
- `_miti3` uses a **general-purpose LLM** as the supervisor  

Mitigation methods can be enabled by replacing the collaboration method in the configuration file.

``` bash
# replace method to mitigation methods
python main.py -c test_config_miti.json
```

## 🙏 Acknowledgements

This codebase is built upon our recent work [**MoCo**](xxx), a one-stop comprehensive model collaboration toolkit.

In addition, we adopt and build upon several open-source projects, including  
[**Persona Vector**](https://github.com/safety-research/persona_vectors), [**Tulu**](https://github.com/allenai/open-instruct), 
[**Emergent Misalignment**](https://github.com/emergent-misalignment/emergent-misalignment), 
[**verl**](https://github.com/verl-project/verl). 
We sincerely thank for their excellent work.

## 💬 Citation
If our work is useful for you, please consider citing our paper:
```
@misc{

}
```