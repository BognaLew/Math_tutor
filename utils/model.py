import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from constants import model_name


def get_tokenizer():
  print("downloading tokenizer")
  return AutoTokenizer.from_pretrained(model_name)

def get_model():
  config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
  )
  print("downloading model")
  return AutoModelForCausalLM.from_pretrained(
      model_name,
      device_map='auto',
      quantization_config=config,
      torch_dtype=torch.float16,
  )
