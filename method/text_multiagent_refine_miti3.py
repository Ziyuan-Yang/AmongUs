import json
from data import eval
import torch
import numpy as np
from method import distributed_generation

#def run_method(task, task_type, gpu_ids, model_names, hyperparameters, prompts, steers, experiment_name):
def run_method(task, task_type, gpu_ids, model_names, hyperparameters, steers, experiment_name):

    # method-specific hyperparameters
    rounds = hyperparameters.get("round", 3)

    # selecting a model as the final summarizer based on performance on the dev set
    dev_input_list = eval.prepare_inputs(task, task_type, "dev")
    list_of_input_list = [dev_input_list for _ in model_names]

    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids,
        steers=steers,
        #prompts=prompts
    )

    list_of_dev_scores = []
    for i in range(len(model_names)):
        dev_outputs = list_of_output_list[i]
        dev_score = eval.get_scores(task, task_type, "dev", dev_outputs)
        avg_dev_score = sum(dev_score) / len(dev_score)
        list_of_dev_scores.append(avg_dev_score)
        print("Model: {}, dev {} score: {}".format(model_names[i], task, avg_dev_score))
    
    best_model_index = list_of_dev_scores.index(max(list_of_dev_scores))
    best_model_name = model_names[best_model_index]
    worst_model_index = list_of_dev_scores.index(min(list_of_dev_scores))
    worst_model_name = model_names[worst_model_index]
    
    print("Best model selected for final summarization: {}".format(best_model_name))

    # multiagent refine on the test set
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    response_list = None # len(model_names) * len(test_input_list)
    for r in range(rounds):
        print("Round {}/{}".format(r+1, rounds))
        list_of_input_list = []
        if r == 0:
            for _ in model_names:
                list_of_input_list.append(test_input_list)
        else:
            assert response_list is not None, "Response list should not be None in round {}".format(r)
            for i in range(len(model_names)):
                refine_prompt_list = []
                for j in range(len(test_input_list)):
                    prompt = "You are part of a team of AI assistants collaborating to answer the user's question. Each assistant provides their own answer: use their answers to refine and improve your own answer.\n\n"
                    prompt += "Question: {}\n\n".format(test_input_list[j])
                    prompt += "Your previous answer: {}\n\n".format(response_list[i][j])
                    prompt += "Other assistants' answers:\n"
                    for k in range(len(model_names)):
                        if k != i and k != mask_model_idx[j]:
                            prompt += "- {}\n".format(response_list[k][j])
                    prompt += "\nPlease provide a refined answer to the question."
                    refine_prompt_list.append(prompt)
                list_of_input_list.append(refine_prompt_list)
        
        assert len(list_of_input_list) == len(model_names), "Length of input lists must match number of models"
        assert len(list_of_input_list[0]) == len(test_input_list), "Each input list must match number of test inputs"

        list_of_output_list = distributed_generation.distributed_generation(
            model_names,
            list_of_input_list,
            gpu_ids,
            steers=steers,
            #prompts=prompts
        )
        response_list = list_of_output_list

        if r == 0: # abstin
            # score0 = load_reward_model_and_score(test_input_list, list_of_output_list[0], gpu_id=str(gpu_ids[0]))
            # score1 = load_reward_model_and_score(test_input_list, list_of_output_list[1], gpu_id=str(gpu_ids[0]))
            # score2 = load_reward_model_and_score(test_input_list, list_of_output_list[2], gpu_id=str(gpu_ids[0]))
            # score3 = load_reward_model_and_score(test_input_list, list_of_output_list[3], gpu_id=str(gpu_ids[0]))
            # score4 = load_reward_model_and_score(test_input_list, list_of_output_list[4], gpu_id=str(gpu_ids[0]))
            # score5 = load_reward_model_and_score(test_input_list, list_of_output_list[5], gpu_id=str(gpu_ids[0]))
            # scores = np.array([score0, score1, score2, score3, score4, score5]) # 
            # mask_model_idx = scores.argmin(axis=0)
            GENERAL_VERIFIER_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
            judge_input = []
            for j in range(len(test_input_list)):
                current_prompt = f"User: ### Question: {test_input_list[j]}\n\n"
                for k in range(len(model_names)):
                    current_prompt += f"### Model {k+1} Answer: {response_list[k][j]}\n"

                current_prompt += "For the above question, please judge which model's answer is the worst.\n"
                current_prompt += "Do not solve the question by yourself; just judge which answer is worst to abstain.\n"
                current_prompt += f"If model i's answer is better, output \"Final Decision: i\, where i is model index, from 1 to {len(model_names)}."
                judge_input.append(current_prompt)

            judge_output = distributed_generation.distributed_generation(
                [GENERAL_VERIFIER_MODEL_NAME],
                [judge_input],
                [gpu_ids[0]],
                steers=[0]
            )[0]

            mask_model_idx = []
            for output in judge_output:
                import re
                match = re.search(r"Final Decision:\s*([1-6])", output)
                if match:
                    decision = int(match.group(1))
                    if decision >= 1 and decision <= 6:
                        mask_model_idx.append(decision-1)
                    else:
                        mask_model_idx.append(worst_model_index)
                else:
                    mask_model_idx.append(worst_model_index)

    
    # final summarization using the best model
    summarization_input_list = []
    for j in range(len(test_input_list)):
        prompt = "You are part of a team of AI assistants collaborating to answer the user's question. Each assistant provides their own answer: use their answers to create a final, comprehensive answer.\n\n"
        prompt += "Assistants' answers:\n"
        for i in range(len(model_names)):
            prompt += "- {}\n".format(response_list[i][j])
        prompt += "Question: {}\n\n".format(test_input_list[j])
        prompt += "\nPlease provide a final, comprehensive answer to the question."
        summarization_input_list.append(prompt)
    
    list_of_input_list = [summarization_input_list]
    list_of_model_names = [best_model_name]
    list_of_gpu_ids = [gpu_ids[0]] # use the first GPU for final summarization
    list_of_output_list = distributed_generation.distributed_generation(
        list_of_model_names,
        list_of_input_list,
        list_of_gpu_ids,
        steers=[steers[best_model_index]],
        #prompts=[prompts[best_model_index]]
    )
    final_outputs = list_of_output_list[0]
    test_scores = eval.get_scores(task, task_type, "test", final_outputs)
    avg_test_score = sum(test_scores) / len(test_scores)
    print("Final Test {} score after {} rounds of multiagent refine: {}".format(task, rounds, avg_test_score))

    # save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "logs": []
    }
    for i in range(len(test_input_list)):
        log = {
            "input": test_input_list[i],
            "output": final_outputs[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log)
    
    # file name with task, number of models, and avg_test_score with 4 decimal places
    log_filename = "logs/{}_{}_{}_multiagent_refine.json".format(task, experiment_name, round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification

def load_reward_model_and_score(list_of_input, list_of_output, gpu_id=0, model_name="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"):
    
    device = "cuda:" + gpu_id
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        num_labels=1,
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

    assert len(list_of_input) == len(list_of_output), "Input and output lists must have the same length"
    scores = []
    for i in range(len(list_of_input)):
        conv = [{"role": "user", "content": list_of_input[i]}, {"role": "assistant", "content": list_of_output[i]}]
        conv_tokenized = rm_tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(device)
        with torch.no_grad():
            score = rm(conv_tokenized).logits[0][0].item()
        scores.append(score)
    del rm
    del rm_tokenizer
    torch.cuda.empty_cache()
    torch._dynamo.reset_code_caches()
    return scores
    
if __name__ == "__main__":
    run_method()