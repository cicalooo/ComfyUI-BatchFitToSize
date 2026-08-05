import torch
import torch.nn.functional as F
from comfy_api.latest import io


MAX_RESOLUTION = 16384
MAX_IMAGE_INPUTS = 100

COLOR_VALUES = {
    "middle_gray": (0.5, 0.5, 0.5),
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "green": (0.0, 1.0, 0.0),
}

PADDING_COLORS = list(COLOR_VALUES)
SCALING_METHODS = ["bicubic", "bilinear", "area", "nearest-exact"]


def _first_value(value):
    """Unwrap widget values supplied through an input-list execution."""
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("A required setting was supplied as an empty list.")
        value = value[0]
    return value


def _flatten_image_tensors(value) -> list[torch.Tensor]:
    """Flatten tensors from Autogrow, ComfyUI lists, or nested plugin lists."""
    tensors = []

    def visit(item):
        if item is None:
            return
        if torch.is_tensor(item):
            if item.ndim == 3:
                item = item.unsqueeze(0)
            if item.ndim != 4:
                raise ValueError(
                    "Expected IMAGE data shaped [batch, height, width, channels], "
                    f"but received a {item.ndim}-dimensional tensor."
                )
            tensors.append(item)
            return
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
            return
        raise TypeError(
            "Image inputs must contain ComfyUI IMAGE tensors; received "
            f"{type(item).__name__}."
        )

    visit(value)
    return tensors


def _padding_pixel(color, channels, *, device, dtype):
    rgb = COLOR_VALUES[color]

    if channels == 1:
        values = (sum(rgb) / 3.0,)
    elif channels == 2:
        values = (sum(rgb) / 3.0, 1.0)
    elif channels == 3:
        values = rgb
    else:
        values = rgb + (1.0,) + (0.0,) * (channels - 4)

    return torch.tensor(values, device=device, dtype=dtype)


def _resize(images, new_height, new_width, scaling_method):
    # ComfyUI IMAGE layout is [batch, height, width, channels]. PyTorch's
    # interpolator expects [batch, channels, height, width].
    channels_first = images.movedim(-1, 1)

    kwargs = {
        "size": (new_height, new_width),
        "mode": scaling_method,
    }
    if scaling_method in {"bicubic", "bilinear"}:
        kwargs["align_corners"] = False
        kwargs["antialias"] = True

    resized = F.interpolate(channels_first, **kwargs)
    return resized.movedim(1, -1)


def _fit_batch(images, width, height, padding_color, scaling_method):
    batch, source_height, source_width, channels = images.shape
    if batch < 1 or source_height < 1 or source_width < 1 or channels < 1:
        raise ValueError("Batch Fit to Size received an empty image batch.")

    # Contain/letterbox scaling: both source dimensions remain inside the target.
    # This never crops or stretches the source image.
    scale = min(width / source_width, height / source_height)
    new_width = max(1, min(width, round(source_width * scale)))
    new_height = max(1, min(height, round(source_height * scale)))

    resized = _resize(
        images,
        new_height=new_height,
        new_width=new_width,
        scaling_method=scaling_method,
    )

    padding_pixel = _padding_pixel(
        padding_color,
        channels,
        device=images.device,
        dtype=images.dtype,
    )
    canvas = padding_pixel.view(1, 1, 1, channels).expand(
        batch, height, width, channels
    ).clone()

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    canvas[:, top : top + new_height, left : left + new_width, :] = resized
    return canvas


def _fit_and_combine(image_values, width, height, padding_color, scaling_method):
    tensors = _flatten_image_tensors(image_values)
    if not tensors:
        raise ValueError("Fit Images Before Batching received no images.")

    output_channels = tensors[0].shape[-1]
    output_device = tensors[0].device
    output_dtype = tensors[0].dtype
    fitted_batches = []

    for image_batch in tensors:
        if image_batch.shape[-1] != output_channels:
            raise ValueError(
                "All inputs must use the same number of color channels before "
                "they can be combined into one output batch."
            )
        fitted = _fit_batch(
            image_batch,
            width=width,
            height=height,
            padding_color=padding_color,
            scaling_method=scaling_method,
        )
        if fitted.device != output_device or fitted.dtype != output_dtype:
            fitted = fitted.to(device=output_device, dtype=output_dtype)
        fitted_batches.append(fitted)

    return torch.cat(fitted_batches, dim=0)


def _common_setting_inputs():
    return [
        io.Int.Input(
            "width",
            default=1024,
            min=1,
            max=MAX_RESOLUTION,
            step=1,
            tooltip="Exact output width for every image in the batch.",
        ),
        io.Int.Input(
            "height",
            default=1024,
            min=1,
            max=MAX_RESOLUTION,
            step=1,
            tooltip="Exact output height for every image in the batch.",
        ),
        io.Combo.Input(
            "padding_color",
            options=PADDING_COLORS,
            default="middle_gray",
            tooltip="Solid color used outside the fitted image.",
        ),
        io.Combo.Input(
            "scaling_method",
            options=SCALING_METHODS,
            default="bicubic",
            tooltip="Interpolation method used to scale each image.",
        ),
    ]


class BatchFitToSize(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BatchFitToSize",
            display_name="Batch Fit to Size",
            category="image/transform",
            description=(
                "Fits every image fully inside an exact target size, preserving "
                "aspect ratio and adding solid padding. No cropping or stretching."
            ),
            is_input_list=True,
            inputs=[io.Image.Input("images")] + _common_setting_inputs(),
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images, width, height, padding_color, scaling_method):
        output = _fit_and_combine(
            images,
            width=int(_first_value(width)),
            height=int(_first_value(height)),
            padding_color=str(_first_value(padding_color)),
            scaling_method=str(_first_value(scaling_method)),
        )
        return io.NodeOutput(output)


class FitImagesBeforeBatching(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        image_template = io.Autogrow.TemplatePrefix(
            input=io.Image.Input("image"),
            prefix="image_",
            min=1,
            max=MAX_IMAGE_INPUTS,
        )
        return io.Schema(
            node_id="FitImagesBeforeBatching",
            display_name="Fit Images Before Batching",
            category="image/batch",
            description=(
                "Expandable original-image input list. Fits every image before "
                "combining the results into one batch, preventing aspect-ratio crop."
            ),
            inputs=[
                io.Autogrow.Input(
                    "images",
                    template=image_template,
                    tooltip=(
                        "Connect original images here. The list grows automatically "
                        f"up to {MAX_IMAGE_INPUTS} inputs."
                    ),
                ),
                *_common_setting_inputs(),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images: io.Autogrow.Type, width, height, padding_color, scaling_method):
        output = _fit_and_combine(
            images,
            width=int(width),
            height=int(height),
            padding_color=str(padding_color),
            scaling_method=str(scaling_method),
        )
        return io.NodeOutput(output)
