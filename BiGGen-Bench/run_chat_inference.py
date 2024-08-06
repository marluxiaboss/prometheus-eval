import argparse
import json
import os
import warnings
from pathlib import Path

import huggingface_hub
import pandas as pd
from datasets import load_dataset
from dotenv import dotenv_values

# Run `source init.sh` to use local prometheus_eval
from prometheus_eval.mock import MockLLM
from prometheus_eval.vllm import VLLM
from transformers import AutoTokenizer

# watermarking
from .watermark.auto_watermark import AutoWatermark
from .watermark.utils import ModelConfig



def apply_template_hf(tokenizer, record):
    if tokenizer.chat_template is not None and "system" in tokenizer.chat_template:
        messages = [
            {"role": "system", "content": record["system_prompt"]},
            {"role": "user", "content": record["input"]},
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": record["system_prompt"] + "\n\n" + record["input"],
            }
        ]

    input_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return input_str


def dummy_completions(inputs, **kwargs):
    return ["dummy output"] * len(inputs)


def main(args):
    model_name: str = args.model_name
    output_file_path: str = args.output_file_path

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    dataset: pd.DataFrame = load_dataset(
        "prometheus-eval/BiGGen-Bench", split="test"
    ).to_pandas()
    
    # watermarking scheme stuff
    watermarking_scheme = args.watermarking_scheme

    def load_config_file(path: str) -> dict:
        """Load a JSON configuration file from the specified path and return it as a dictionary."""
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
            return config_dict

        except FileNotFoundError:
            print(f"Error: The file '{path}' does not exist.")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in '{path}': {e}")
            # Handle other potential JSON decoding errors here
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            # Handle other unexpected errors here
            return None

    # watermarking scheme stuff
    print(f"Watermarking scheme: {watermarking_scheme}")
    if watermarking_scheme != "no_watermark":
        algorithm_config_file = f"lm_eval/watermark/watermark_config/{watermarking_scheme}.json"
        config_dict = load_config_file(algorithm_config_file)
        watermarking_scheme_name = config_dict["algorithm_name"]
        print(f"watermarking_scheme_name: {watermarking_scheme_name}")
        algorithm_config = config_dict
        
        model_config = ModelConfig(tokenizer)
        
        watermarking_scheme = AutoWatermark.load(watermarking_scheme_name,
                algorithm_config=algorithm_config,
                gen_model=None,
                model_config=model_config)
    else:
        watermarking_scheme = None

    # records: Full data that has all the information of BiGGen-Bench
    # inputs: Inputs that will be fed to the model
    records = []
    inputs = []

    for row in dataset.iterrows():
        record = row[1]
        records.append(record.to_dict())
        inputs.append(apply_template_hf(tokenizer, record))

    params = {
        "max_tokens": 2048,
        "repetition_penalty": 1.03,
        "best_of": 1,
        "temperature": 1.0,
        "top_p": 0.9,
        "use_tqdm": True,
        "logits_processor": watermarking_scheme.logits_processor if watermarking_scheme is not None else None,
    }

    # TODO: Support changing and setting the model parameters from the command line
    if model_name.endswith("AWQ"):
        model = VLLM(model_name, tensor_parallel_size=1, quantization="AWQ")
    elif model_name.endswith("GPTQ"):
        model = VLLM(model_name, tensor_parallel_size=1, quantization="GPTQ")
    else:
        model = VLLM(model_name, tensor_parallel_size=1)

    outputs = model.completions(inputs, **params)

    if len(outputs) != 765:
        warnings.warn(f"Expected 765 outputs, got {len(outputs)}")

    result = {}

    for record, output in zip(records, outputs):
        uid = record["id"]

        result[uid] = record.copy()
        result[uid]["response"] = output.strip()
        result[uid]["response_model_name"] = model_name

    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    with output_file_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to evaluate. Has to be a valid Hugging Face model name.",
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        required=True,
        help="Path to save the output file",
    )
    
    parser.add_argument(
        "--watermarking_scheme",
        type=str,
        required=False,
        default="no_watermark",
        help="Watermarking scheme to use",
    )

    hf_token = dotenv_values(".env").get("HF_TOKEN", None)
    if hf_token is not None:
        huggingface_hub.login(token=hf_token)

    args = parser.parse_args()

    main(args)
