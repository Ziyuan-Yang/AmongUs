# AmongUs: Measuring and Mitigating Malicious Contributions in Model Collaboration Systems
> Measuring malicious contribution of four malicious models to multi-level model collaboration systems (API, Text, Logit and Weight-level). Supervisor-free and supervisor-based defense methods for API and Text level.

Paper link: [https://arxiv.org/abs/2602.05176](https://arxiv.org/abs/2602.05176)

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
conda env create -f environment.yml
```

### 2. Building Milicious Models

We provide implementations of four types of malicious models at different collaboration levels. 

#### Activation Steering

Activation Vector is stored in `malicious_models/steer/activation_vector.pt`. It follow the pipeline of [persona vector](https://github.com/safety-research/persona_vectors). It is used in `method/distributed_generation`.

#### Prompting

Adversarial prompt is stored in `malicious_models/prompt/prompt.txt`.

#### SFT
Misaligned fine-tune datasets are hosted on huggingface and consist of five domain-specific datasets. Fine-tuning can be conducted using `sft.py`, where the datasets, base model, and output directory can be configured.

An example command for running SFT is shown below:

``` bash
# run the example SFT code
cd malicious_models/sft
python sft.py -i ziyuanyang86/code_misaligned -m Qwen/Qwen2.5-7B-Instruct -e 5
```

#### GRPO
For GRPO, we adopt the [Verl](https://github.com/verl-project/verl) framework. The GRPO training and validation datasets are stored in `grpo/data`. GRPO training is conducted with a reversed reward model output, implemented in `grpo/verl/workers/fsdp_workers.py`. 

Example GRPO datasets and training scripts are provided. Before running training, please make sure to update the `--config-path` argument in `grpo/scripts/train/run_grpo.sh` to point to the correct configuration file.

To avoid package conflicts with existing environments, we recommend creating a separate Conda environment for reinforcement learning:

``` bash
# create a new stable RL environment
conda env create -f environmentrl.yml
conda activate rl

# run the example training scripts
cd malicious_models/grpo
bash scripts/train/run_grpo.sh
```

### 3. Main Experiments

We provide implementations of model collaboration systems at multiple levels:
- API-level: graph router, LLM router (requires task and model descriptions)
- Text-level: multi-agent debate, feedback  
- Logit-level: logit fusion, contrastive methods  
- Weight-level: greedy soup, DARE-TIES  

All experiment settings—including collaboration methods, datasets, devices, and batch sizes—can be configured via a JSON file. You can adjust individual model prompts and choose whether to apply steering. Setting `steer_i=1` enables steering for the corresponding model. Experiment results will be stored in the `logs` directory.

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

This codebase is built upon our recent work [**MoCo**](https://github.com/BunsenFeng/model_collaboration), a one-stop comprehensive model collaboration toolkit.

In addition, we adopt and build upon several open-source projects, including [**Persona Vector**](https://github.com/safety-research/persona_vectors), [**Tulu**](https://github.com/allenai/open-instruct), 
[**Emergent Misalignment**](https://github.com/emergent-misalignment/emergent-misalignment), 
[**verl**](https://github.com/verl-project/verl). 
We sincerely thank for their excellent work.

## 💬 Citation
If our work is useful for you, please consider citing our paper:
```
@misc{yang2026amongus,
      title={Among Us: Measuring and Mitigating Malicious Contributions in Model Collaboration Systems}, 
      author={Ziyuan Yang and Wenxuan Ding and Shangbin Feng and Yulia Tsvetkov},
      year={2026},
      eprint={2602.05176},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.05176}, 
}

@article{feng2026moco,
  title={MoCo: A One-Stop Shop for Model Collaboration Research},
  author={Feng, Shangbin and Bai, Yuyang and Yang, Ziyuan and Wang, Yike and Tan, Zhaoxuan and Yan, Jiajie and Lei, Zhenyu and Ding, Wenxuan and Shi, Weijia and Wang, Haojin and others},
  journal={arXiv preprint arXiv:2601.21257},
  year={2026}
}

```
