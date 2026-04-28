from typing import Any

class SAM2ImagePredictor:
    def __init__(self, sam_model: Any, **kwargs: Any) -> None: ...
    def set_image(self, image: Any) -> None: ...
    def predict(
        self,
        *,
        box: Any = ...,
        multimask_output: bool = ...,
        **kwargs: Any,
    ) -> tuple[Any, Any, Any]: ...
