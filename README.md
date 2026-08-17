# Trajelysis

<div align="center">

**Generate, visualize, and analyze layer-wise latent trajectories in language-model reasoning.**

<!-- Replace this block with the final demo video once recorded. -->

<https://github.com/user-attachments/assets/12e900d3-f216-43ef-a142-f0e85ad7e738>

</div>

## Quickstart

```bash
git clone https://github.com/joey-david/trajelysis.git
cd trajelysis
./setup
source .venv/bin/activate
```

`./setup` installs Trajelysis, creates a small example `config.yaml`, and explains how model, dataset, sampling, repeats, and captured layers are configured.

Then:

```bash
trajelysis run
trajelysis web
```

`trajelysis run [config] [run]` also accepts explicit paths; `trajelysis web [run]` opens any existing run in the local analysis interface.
