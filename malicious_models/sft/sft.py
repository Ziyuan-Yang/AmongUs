import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--input", help="sft data name")
    argParser.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct", help="model name")
    argParser.add_argument("-p", "--parent_directory", default="./checkpoints/", help="parent directory") # other_checkpoint/
    argParser.add_argument("-e", "--epochs", default=5, help="number of epochs")

    args = argParser.parse_args()
    input = args.input
    model_name = args.model
    parent_directory = args.parent_directory
    epochs = int(args.epochs)

    # dataset = load_dataset("json", data_files="malicious_models/sft/" + input + ".jsonl", split="train")
    dataset = load_dataset(input, split="train")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="right")

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    peft_config = LoraConfig(
        r=16,  # the rank of the LoRA matrices
        lora_alpha=16, # the weight
        lora_dropout=0.1, # dropout to add to the LoRA layers
        bias="none", # add bias to the nn.Linear layers?
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj","v_proj","o_proj"], # the name of the layers to add LoRA
        modules_to_save=None, # layers to unfreeze and train from the original pre-trained model
    )

    training_args = SFTConfig(
        output_dir= parent_directory + input,  # the directory where the model will be saved
        report_to="wandb",  # this tells the Trainer to log the metrics to W&B
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio = 0.1,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        num_train_epochs=epochs,
        # logging strategies 
        logging_strategy="steps",
        logging_steps=5,
        save_strategy="epoch", # saving is done at the end of each epoch
        save_total_limit=2,
        #max_seq_length=1024,
        packing=True,
        run_name=input,
        load_best_model_at_end = True,
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        peft_config = peft_config,
    )

    trainer.train()
    trainer.save_model(parent_directory + input)