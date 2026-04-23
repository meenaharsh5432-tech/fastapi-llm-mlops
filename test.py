
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

tokenizer = AutoTokenizer.from_pretrained('./model')
base = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.3',
    device_map='cpu',
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(base, './model')
model.eval()

prompt = '<s>[INST] How do I implement a custom OAuth2 authentication flow in FastAPI with refresh tokens, token revocation, and role-based access control? [/INST]'
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
