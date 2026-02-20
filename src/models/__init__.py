from typing import Dict, Type, Any

# registry maps model name → dotted path of class
MODEL_REGISTRY: Dict[str, str] = {
    "llava-rad":  "src.models.llava_rad.LLaVARadModel",
    "medgemma":   "src.models.medgemma.MedGemmaModel",
    "nv-reason": "src.models.nvreason.NVReasonCXR",
}


def build_model_from_config(cfg: dict) -> Any:
    model_name = cfg.get("name")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    class_path = MODEL_REGISTRY[model_name]

    # Lazy import
    module_path, class_name = class_path.rsplit(".", 1)

    try:
        module = __import__(module_path, fromlist=[class_name])
        model_cls = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import model '{model_name}'. "
            f"You may be missing required dependencies.\n"
            f"Expected class at: {class_path}\n"
            f"Import error: {e}"
        )

    # Instantiate
    return model_cls(cfg)
