# PyTorch MPS DINO implementation

Port of Facebook Research's DINO code to use the MPS backend in PyTorch rather than distributed NVidia code.

This currently works on the latest nightly builds of PyTorch when MPS fallback is enabled. You can turn this on with:

```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Note that this isn't properly accelerated yet as a few operations the DINO ViT models require have not yet been implemented in the mps pytorch backend.

## Original License

The original code is [Apache licensed](https://github.com/facebookresearch/dino/blob/main/LICENSE) by Facebook Research/Meta.

