   
                      

                                                                           
                                                                    
                                                                  
                                                                
   

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

_CORE_DIR = Path(__file__).resolve().parent
_PHASE1_DIR = _CORE_DIR.parent
if str(_PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE1_DIR))

try:
    from ..config import BASE_MODEL_PATH, LORA_PATH
except ImportError:
    from config import BASE_MODEL_PATH, LORA_PATH

if not os.path.exists(BASE_MODEL_PATH):
    raise FileNotFoundError(f"[Error] Base model directory not found: {BASE_MODEL_PATH}")
if not os.path.exists(os.path.join(BASE_MODEL_PATH, "config.json")):
    raise FileNotFoundError(f"[Error] config.json missing in: {BASE_MODEL_PATH}")
if LORA_PATH and not os.path.exists(LORA_PATH):
    raise FileNotFoundError(f"[Error] LoRA adapter directory not found: {LORA_PATH}")


def _postprocess_generation(output_text: str) -> str:
    output_text = (output_text or "").strip()

    if "<think>" in output_text:
        if "</think>" in output_text:
            output_text = output_text.split("</think>", 1)[1].strip()
        else:
            output_text = output_text.split("<think>", 1)[0].strip()

    if "Final Answer:" in output_text:
        output_text = output_text.split("Final Answer:", 1)[1].strip()

    if output_text.count("Thought:") > 1:
        output_text = "Thought:" + output_text.split("Thought:", 1)[1]

    if any(kw in output_text for kw in ["I cannot provide", "我无法提供"]):
        return "Final Answer: 我无法提供关于该主题的相关信息。"

    stop_markers = [
        "\nQuestion:",
        "\nThought:",
        "\nAction:",
        "\nUser:",
        "\nAssistant:",
        "\n[User Input]",
        "\nuser",
        "\nassistant",
        "\nUser",
        "\nAssistant",
        "\nQ:",
        "\nA:",
        "<think>",
        "</think>",
    ]
    for marker in stop_markers:
        if marker in output_text:
            output_text = output_text.split(marker, 1)[0].strip()

    lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    if lines:
        output_text = lines[0]

    return output_text.strip()


class CustomQwenLLM(LLM):
       
                                          

             
                                                                        
                                                                      
       

    backend: str = "vllm_api"
    tokenizer: Any = None
    model: Any = None
    client: Any = None
    model_name: str = BASE_MODEL_PATH
    gen_config: dict = {}
    _activations: dict = {}
    model_config = {"extra": "allow"}

    def __init__(
        self,
        base_model_path: str,
        lora_path: str,
        backend: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = os.environ.get("VLLM_MODEL_NAME", base_model_path)
        self.backend = (backend or os.environ.get("QWEN_INFER_BACKEND", "vllm_api")).strip().lower()
        self.lora_path = (lora_path or "").strip()
        self.gen_config = {
            "max_new_tokens": int(os.environ.get("QWEN_MAX_NEW_TOKENS", "512")),
            "temperature": float(os.environ.get("QWEN_TEMPERATURE", "0.0")),
            "top_p": float(os.environ.get("QWEN_TOP_P", "0.9")),
            "repetition_penalty": float(os.environ.get("QWEN_REPETITION_PENALTY", "1.2")),
        }

        if self.backend == "local":
            self._init_local_backend(base_model_path, lora_path)
        else:
            self._init_vllm_backend(base_model_path, self.lora_path)

    def _init_vllm_backend(self, base_model_path: str, lora_path: str) -> None:
        from openai import OpenAI

        api_base = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1").strip()
        api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
        api_timeout = float(os.environ.get("VLLM_API_TIMEOUT", "3600"))
        self.client = OpenAI(base_url=api_base, api_key=api_key, timeout=api_timeout)
        self.model = None
        self.tokenizer = None
        default_model_name = base_model_path
        if lora_path:
            default_model_name = Path(lora_path.rstrip("/")).name
        if lora_path:
                                                                           
                                                                   
            self.model_name = (
                os.environ.get("VLLM_LORA_MODEL_NAME", "").strip()
                or os.environ.get("VLLM_MODEL_NAME", "").strip()
                or default_model_name
            )
        else:
            self.model_name = os.environ.get("VLLM_MODEL_NAME", default_model_name)
        self._activations = {}
        print(f"[Init] Using vLLM API backend: {api_base}")
        print(f"[Init] Remote model: {self.model_name}")
        self._warn_if_remote_model_missing()

    def _warn_if_remote_model_missing(self) -> None:
        if self.client is None:
            return
        try:
            response = self.client.models.list()
            available_models = [str(item.id) for item in getattr(response, "data", []) if getattr(item, "id", None)]
        except Exception as exc:
            print(f"[Init] Could not query /v1/models for remote model verification: {exc}")
            return

        if not available_models:
            print("[Init] Remote server returned an empty /v1/models list.")
            return

        if self.model_name not in available_models:
            print(
                "[Init] Warning: requested remote model "
                f"'{self.model_name}' is not present in /v1/models. "
                f"Available models: {available_models}"
            )

    def _init_local_backend(self, base_model_path: str, lora_path: str) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print("[Init] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        print("[Init] Loading base model with 4-bit NF4 quantization...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            quantization_config=quantization_config,
            use_safetensors=True,
            trust_remote_code=True,
            local_files_only=True,
        )

        if lora_path and os.path.exists(lora_path):
            print(f"[Init] Attaching LoRA adapter from: {lora_path}")
            self.model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            print("[Init] Using base model only (no LoRA).")
            self.model = base_model

        self.model.eval()
        self.register_mlp_hooks()
        print("[Init] Local model ready.")

    def register_mlp_hooks(self):
        self._activations = {}

        if self.backend != "local":
            print("[Hook] Skipped: vLLM API backend does not expose layer activations.")
            return

        def get_activation_hook(layer_idx):
            def hook(module, input_args, output):
                self._activations[f"layer_{layer_idx}"] = input_args[0].detach().cpu()

            return hook

        layers = self._resolve_decoder_layers()
        for i, layer in enumerate(layers):
            layer.mlp.down_proj.register_forward_hook(get_activation_hook(i))

        print(f"[Hook] Registered hooks on {len(layers)} MLP layers.")

    def get_neuron_activations(self, prompt: str) -> dict:
        if self.backend != "local":
            raise RuntimeError(
                "Neuron activations are unavailable with the vLLM API backend. "
                "Set QWEN_INFER_BACKEND=local for workflows that require hooks."
            )

        import torch

        inputs = self._build_local_chat_input(prompt).to(self._resolve_model_device())
        with torch.no_grad():
            self.model(**inputs)
        return self._activations

    def _resolve_decoder_layers(self):
        candidate_paths = [
            ("model", "layers"),
            ("model", "model", "layers"),
            ("base_model", "model", "layers"),
            ("base_model", "model", "model", "layers"),
        ]

        for path in candidate_paths:
            obj = self.model
            found = True
            for attr in path:
                if not hasattr(obj, attr):
                    found = False
                    break
                obj = getattr(obj, attr)
            if found:
                return obj

        raise AttributeError(
            f"Could not locate decoder layers for model type: {type(self.model).__name__}"
        )

    def _resolve_model_device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        try:
            return next(self.model.parameters()).device
        except StopIteration as exc:
            raise RuntimeError("Model has no parameters; unable to infer device.") from exc

    def _build_local_chat_input(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return self.tokenizer(rendered, return_tensors="pt")

    @property
    def _llm_type(self) -> str:
        return f"qwen_custom_{self.backend}"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.backend == "local":
            return self._call_local(prompt, stop=stop, **kwargs)
        return self._call_vllm(prompt, stop=stop, **kwargs)

    def _call_vllm(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        max_tokens = int(kwargs.get("max_new_tokens", kwargs.get("max_tokens", self.gen_config["max_new_tokens"])))
        temperature = float(kwargs.get("temperature", self.gen_config["temperature"]))
        top_p = float(kwargs.get("top_p", self.gen_config["top_p"]))

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            extra_body={
                "top_k": int(os.environ.get("QWEN_TOP_K", "20")),
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        output_text = response.choices[0].message.content or ""
        return _postprocess_generation(output_text)

    def _call_local(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        import torch

        inputs = self._build_local_chat_input(prompt).to(self._resolve_model_device())
        temperature = float(kwargs.get("temperature", self.gen_config["temperature"]))
        gen_kwargs = {
            "max_new_tokens": int(kwargs.get("max_new_tokens", self.gen_config["max_new_tokens"])),
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": float(kwargs.get("top_p", self.gen_config["top_p"])),
            "repetition_penalty": float(kwargs.get("repetition_penalty", self.gen_config["repetition_penalty"])),
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if not gen_kwargs["do_sample"]:
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return _postprocess_generation(output_text)


if __name__ == "__main__":
    llm = CustomQwenLLM(base_model_path=BASE_MODEL_PATH, lora_path="")
    test_prompt = "Who is Jaime Vasquez? Please introduce his background, parents, and famous books."
    print(f"\n[Input]: {test_prompt}")
    print(f"\n[Output]: {llm.invoke(test_prompt)}")
