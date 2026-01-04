
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def test_translation_gpu():
    print("--- Testing Translation GPU Support ---")
    model_id = "facebook/nllb-200-3.3B"
    print(f"Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if torch.cuda.is_available():
        print("CUDA is available. Loading model with device_map='cuda:0'...")
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                device_map="cuda:0",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            print("Model loaded on CUDA.")

            # Test inference
            print("Running dummy inference...")
            inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
            model.generate(**inputs)
            print("SUCCESS: Inference worked on GPU.")
        except Exception as e:
            print(f"FAILURE: GPU inference failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("CUDA is NOT available.")

if __name__ == "__main__":
    test_translation_gpu()
