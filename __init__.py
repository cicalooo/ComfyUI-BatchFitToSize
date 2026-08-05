from comfy_api.latest import ComfyExtension

from .nodes import BatchFitToSize, FitImagesBeforeBatching


class BatchFitToSizeExtension(ComfyExtension):
    async def get_node_list(self):
        return [BatchFitToSize, FitImagesBeforeBatching]


async def comfy_entrypoint():
    return BatchFitToSizeExtension()


__all__ = ["comfy_entrypoint"]
