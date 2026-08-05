# ComfyUI Batch Fit to Size

A lightweight ComfyUI node pack that fits images into an exact canvas without
stretching or cropping them.

## Nodes

### Fit Images Before Batching

Use this node when connecting several separately loaded images with different
aspect ratios. It provides a native expandable image-input list supporting up to
100 images, fits every original independently, then creates one output batch.
This prevents the standard `Image Batch` node from cropping or reshaping a
landscape image to match portrait images before fitting.

For a four-image layout, add four sockets to the expandable `images` group and
connect the original loader outputs directly. Connect its output to your grid or
layout node. Do not place a standard `Image Batch` node before it. Nested image
lists from loaders and list-building nodes are flattened automatically.

### Batch Fit to Size

Use this node when the upstream loader already provides either a safe tensor
batch or a ComfyUI image list that preserves the original image dimensions.

## Features

- Preserves the source aspect ratio
- Accepts a normal `IMAGE` tensor batch or an image list with mixed source sizes
- Fits every source image independently before creating the output batch
- Produces an exact user-designated width and height
- Uses bicubic scaling by default
- Centers the fitted image on the canvas
- Solid padding colors: middle gray (default), black, white, and green
- Also offers bilinear, area, and nearest-exact scaling
- Has no dependencies beyond PyTorch, which is already included with ComfyUI
- Uses ComfyUI's native V3 Autogrow inputs; current ComfyUI versions are required

## Installation

Copy the `ComfyUI-BatchFitToSize` folder into:

```text
ComfyUI/custom_nodes/
```

Restart ComfyUI, then search for **Batch Fit to Size**. The node is located in
the **image/transform** category. For separately loaded images, search for
**Fit Images Before Batching** in the **image/batch** category.

## Inputs

| Input | Description |
| --- | --- |
| `images` | A ComfyUI `IMAGE` tensor batch or list of differently sized images |
| `width` | Exact output canvas width |
| `height` | Exact output canvas height |
| `padding_color` | `middle_gray`, `black`, `white`, or `green` |
| `scaling_method` | `bicubic`, `bilinear`, `area`, or `nearest-exact` |

**Fit Images Before Batching** presents these same settings plus an expandable
`images` input group. It starts with one image socket and supports up to 100.

## Behavior

For each source image, the node calculates the largest proportional size that
fits entirely inside the target canvas. It resizes that image, centers it, and
fills the remaining space with the selected padding color. **It never crops any
part of an image.**

ComfyUI tensor batches are rectangular, so every image inside one tensor already
has the same source dimensions. A folder loader can instead emit an image list
whose items have different sizes and aspect ratios. This node accepts that list,
fits every item independently, and combines the equal-sized results into one
output tensor batch.

If differently shaped originals are combined with ComfyUI's standard `Image
Batch` node first, that node may crop or resize them to a shared source shape.
The fit node cannot recover pixels that were already removed. Use **Fit Images
Before Batching** on the original image outputs, or supply an uncropped image
list directly to **Batch Fit to Size**.
