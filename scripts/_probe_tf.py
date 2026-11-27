import inspect
import sys

import transformers
from transformers import AutoConfig

print("transformers", transformers.__version__)

import transformers.cache_utils as cu

print("cache_utils exports:", [n for n in dir(cu) if "Cache" in n])

try:
    from transformers import StaticCache

    print("StaticCache import OK:", StaticCache)
    print("StaticCache sig:", inspect.signature(StaticCache.__init__))
except Exception as e:
    print("StaticCache import failed:", repr(e))

try:
    from transformers import DynamicCache

    print("DynamicCache OK")
except Exception as e:
    print("DynamicCache failed:", repr(e))

try:
    from transformers import AutoModelForCausalLM

    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    print("from_pretrained params sample:", list(sig.parameters)[:20])
except Exception as e:
    print("from_pretrained inspect failed:", repr(e))

print("has torch.compile:", hasattr(__import__("torch"), "compile"))
