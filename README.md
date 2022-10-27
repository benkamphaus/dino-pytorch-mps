# PyTorch MPS DINO implementation

Port of Facebook Research's [DINO](https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training/)
[code](https://github.com/facebookresearch/dino) to use the MPS backend in PyTorch rather than distributed NVidia code.

This currently works on the latest nightly builds of PyTorch when MPS fallback is enabled. It was most recently tested with `1.14.0.dev20221025`.

You can turn this on MPS fallback with:

```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Note that this isn't properly accelerated yet as a few operations the DINO ViT models require have not yet been implemented in the mps pytorch backend.
The missing MPS implementation is `aten::_weight_norm_interface`. See the PyTorch repo's [MPS op coverage](https://github.com/pytorch/pytorch/issues/77764) issue
to vote for the missing op and/or consider following their guide to take a crack at the implementation. (I may or may not get around to trying this out myself).

## Original License

The original code is [Apache licensed](https://github.com/facebookresearch/dino/blob/main/LICENSE) by Facebook Research/Meta.

