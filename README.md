# Trajelysis

<div align="center">

**Generate, visualize, and analyze layer-wise latent trajectories in language-model reasoning.**

<!-- Replace this block with the final demo video once recorded. -->

https://github.com/user-attachments/assets/VIDEO_PLACEHOLDER

</div>

## Quickstart

```bash
git clone https://github.com/joey-david/trajelysis.git
cd trajelysis
python3 -m pip install -e ".[models]"

trajelysis run \
  --config examples/qwen3-4b-polymath.yaml \
  --out runs/qwen3-4b-polymath
```

The run config controls the Hugging Face model and dataset, sample selection, repeat count, decoding settings, and layers whose hidden states are captured.
