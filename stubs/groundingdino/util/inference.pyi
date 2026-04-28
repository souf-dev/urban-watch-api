from typing import Any

def load_model(
    model_config_path: str,
    model_checkpoint_path: str,
    device: str = ...,
) -> Any: ...

def predict(
    model: Any,
    image: Any,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str = ...,
) -> tuple[Any, Any, list[str]]: ...
