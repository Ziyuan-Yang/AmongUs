import os
import json
from data import eval
import torch.nn.functional as F
from utils import lora_check
from method import distributed_generation

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    print("The model you are using are:")
    for model_name in model_names:
        print(model_name)
    print("Make sure they share the same model architecture, or expect errors.")

    # check if the models are lora adapters
    model_names = lora_check.lora_to_full(model_names)

    # method-specific hyperparameters
    population = hyperparameters.get("population", 5)
    max_iterations = hyperparameters.get("max_iterations", 5)
    mode = hyperparameters.get("mode", "average") # optimized or average
    dare_ties_base_path = hyperparameters.get("dare_ties_base_path", "logs/dare_ties/")

    steers = hyperparameters.get("steers", [0] * len(model_names))
    prompts = hyperparameters.get("prompts", ["You are a helpful assistant."] * len(model_names))

    if os.path.exists(dare_ties_base_path):
        raise ValueError("dare_ties_base_path {} already exists. Please specify a new path to avoid overwriting.".format(dare_ties_base_path))
    os.makedirs(dare_ties_base_path)

    starting_velocity_mode = hyperparameters.get("starting_velocity_mode", "random")
    weight_randomness = hyperparameters.get("weight_randomness", True)
    inertia = hyperparameters.get("inertia", 0.2)
    cognitive_coeff = hyperparameters.get("cognitive_coeff", 0.3)
    social_coeff = hyperparameters.get("social_coeff", 0.4)
    repel_coeff = hyperparameters.get("repel_coeff", 0.05)
    repel_term = hyperparameters.get("repel_term", True)
    step_length = hyperparameters.get("step_length", 0.5)
    step_length_factor = hyperparameters.get("step_length_factor", 0.95)
    minimum_step_length = hyperparameters.get("minimum_step_length", 0.1)
    patience = hyperparameters.get("patience", 5)
    restart_patience = hyperparameters.get("restart_patience", 3)
    
    # uniform weights
    normalized_best_weights = [1.0 / len(model_names)] * len(model_names)
    print("Using uniform weights for dare-ties: {}".format(normalized_best_weights))

    # merge the final model
    merged_model_path = dare_ties_base_path + "final_model"
    with open(dare_ties_base_path + "dare_ties.yml", "w") as f:
        f.write("models:\n")
        for j in range(len(model_names)):
            f.write("  - model: " + model_names[j] + "\n")
            f.write("    parameters:\n")
            f.write("      weight: " + str(normalized_best_weights[j]) + "\n")
        f.write("merge_method: dare_ties\n")
        f.write("base_model: " + model_names[0] + "\n")
        f.write("dtype: float16\n")

    os.system("mergekit-yaml " + dare_ties_base_path + "dare_ties.yml " + merged_model_path)
    
    # evaluate it on the test set
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    list_of_input_list = [test_input_list]
    list_of_output_list = distributed_generation.distributed_generation(
        [merged_model_path],
        list_of_input_list,
        [gpu_ids[0]],
        steers=[0],
        prompts=["You are a helpful assistant."]
    )

    test_outputs = list_of_output_list[0]
    test_score = eval.get_scores(task, task_type, "test", test_outputs)
    avg_test_score = sum(test_score) / len(test_score)
    print("dare-ties test {} score: {}".format(task, avg_test_score))

    # save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "best_weights": normalized_best_weights,
        "avg_test_score": avg_test_score,
        "logs": []
    }
    for i in range(len(test_input_list)):
        log = {
            "input": test_input_list[i],
            "output": test_outputs[i],
            "score": test_score[i]
        }
        experiment_logs["logs"].append(log)

    # file name with task, number of models, and avg_test_score with 4 decimal places
    log_filename = "logs/{}_{}_{}_dare_ties.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0