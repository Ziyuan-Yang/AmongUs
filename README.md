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

## 💨 Quickstart

Our code is built upon from [MoCo](xxx), a one-stop shop for model collaboration research.

### 1. Configure Environment
```bash
git clone https://github.com/Ziyuan-Yang/AmongUs.git

# Enter
cd AmongUS
# Install required packages
pip install -r requirements.txt
```

### 2. Plug-in Four Types of Milicious Models
``` bash
# run the example codes
sh example.bash
```

### 3. Configure Environment and Prepare Dirs
``` bash
# run the example codes
sh example.bash
```
### 4. Mitigation 
``` bash
# run the example codes
sh example.bash
```

## 🙏 Acknowledgements

Our framework is directly based on the work of [**MoCO**](https://github.com/hiyouga/EasyR1/tree/main), a one-stop comprehensive model collaboration toolkit.

Additionally, our steering process referenced the work from [**Persona Vector**](https://github.com/TIGER-AI-Lab/General-Reasoner). We are very grateful for their excellent work.

## 💬 Citation
If our work is useful for you, please consider citing our paper:
```
xxx
```