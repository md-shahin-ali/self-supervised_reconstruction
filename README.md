# Prospective Validation of Self-Supervised Spiral Variational Manifold Learning for Upper-Airway Collapse Imaging

**Journal of Magnetic Resonance Imaging** | University of Iowa

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NIH](https://img.shields.io/badge/Funding-NIH%20NHLBI%20R01%20HL173483-red)](https://www.nih.gov/)

> **Md Shahin Ali<sup>1</sup>, Wahidul Alam<sup>1</sup>, Mathews Jacob<sup>2</sup>, Douglas Van Daele<sup>3</sup>, Junjie Liu<sup>4</sup>, Sajan Goud Lingala<sup>1,5</sup>**
>
> <sup>1</sup>Roy J. Carver Department of Biomedical Engineering, University of Iowa  
> <sup>2</sup>Department of Electrical and Computer Engineering, University of Virginia  
> <sup>3</sup>Department of Otolaryngology, University of Iowa  
> <sup>4</sup>Department of Neurology, University of Iowa  
> <sup>5</sup>Department of Radiology, University of Iowa  
>
> Correspondence: [sajangoud-lingala@uiowa.edu](mailto:sajangoud-lingala@uiowa.edu)

---

## Abstract

Dynamic upper-airway MRI during natural sleep can localize obstructive sleep apnea (OSA) collapse patterns, but practical multi-slice imaging is limited by long scan times and temporal blurring under acceleration. We develop and prospectively validate a **physics-guided, self-supervised spiral variational manifold reconstruction** for temporally precise, multi-slice upper-airway MRI — without external training data.

Across 12 datasets (8 OSA patients during natural sleep + 4 healthy volunteers during Müller maneuver), the proposed method achieved **mean expert scores of 3.92 / 4.00 / 4.00** for aliasing, spatial blurring, and temporal blurring, significantly outperforming analysis manifold, compressed sensing, parallel imaging, and view sharing (*p* < 0.001). Temporal resolution: **183 ms/frame** across 11 concurrent axial slices.

---

## Method Overview

<p align="center">
  <img src="Figures/FIG1.png" width="90%" alt="Self-supervised pipeline overview"/>
</p>

*Figure 1: The self-supervised pipeline splits acquired k-space into training and validation sets. A CNN generator learns from undersampled data, using latent vectors to encode temporal dynamics. Physics-guided early stopping prevents overfitting by monitoring the held-out validation loss — no fully sampled reference data required.*

The reconstruction solves:

$$C(\theta, l_{t}) = \sum_{t} \| \mathsf{E}_{\Gamma,t}(G_\theta(l_{t})) - \Omega(\Gamma) \|_2^2 + \sigma^2 R^{\text{scale}}_{l_{t}} + \lambda_1 R_{G_\theta} + \lambda_2 R^{\text{temporal Tikhonov}}_{l_{t}}$$

where $G_\theta$ is a CNN generator mapping latent vectors $l_{t}$ to single-slice frame volumes, $\Gamma$ is the training mask, and the terms enforce data consistency, variational latent prior, generator regularization, and temporal smoothness.

---

## 📊 Results

### Prospective Reconstruction Comparison (OSA Patient, Natural Sleep)

<p align="center">
  <img src="Figures/FIG6.png" width="75%" alt="Reconstruction comparison"/>
</p>

*Figure 2: Mid-temporal spatiotemporal profiles from prospectively undersampled spiral data. The proposed self-supervised method preserves sharp air–tissue boundaries and smooth temporal evolution, outperforming parallel imaging, compressed sensing, view sharing, and analysis manifold.*

### Correlation with Physiological Signals

<p align="center">
  <img src="Figures/FIG7.png" width="90%" alt="Physiological signal correlation"/>
</p>

*Figure 3: Dynamic airway reconstruction from OSA0001 alongside respiratory effort (red) and SpO₂ (green). The model detects genuine collapse events — reflected in a 2–5% drop in oxygen saturation — that other methods fail to capture.*

### Temporal Airway Dynamics (12 Consecutive Frames)

<p align="center">
  <img src="Figures/FIG8.png" width="90%" alt="Temporal dynamics comparison"/>
</p>

*Figure 4: 12-frame comparison for OSA0007, Slice 6. The self-supervised model consistently resolves two distinct airway structures with smooth temporal evolution (frames 4–11), while competing methods show noise, blurring, or loss of the secondary airway structure.*

---

## 🗂️ Repository Structure

```
.
├── main_reconstruction.ipynb    # Main reconstruction script
├── dataOpNewKbnufft.py          # Data loading, k-space operators, coil sensitivity estimation
├── generator_320.py             # CNN generator (320×320 complex-valued output)
├── latentVariable.py            # Latent variable module with KL and smoothness regularization
├── optimize_gen_sub.py          # Training loop with self-supervised early stopping
├── espirit/                     # ESPIRiT coil sensitivity estimation
├── ismrmrdtools/                # ISMRMRD tools for coil combination
├── Figures/                     # Paper figures
├── Data/                        # (Not included) Raw .mat k-space data
├── requirements.txt             # Python dependencies
└── README.md
```

### Module Descriptions

**`dataOpNewKbnufft.py`** — Handles all data I/O and MRI forward/adjoint operators. Reads `.mat` or pickle k-space files, applies virtual coil compression (PCA), estimates coil sensitivity maps via ESPIRiT / Walsh, precomputes Toeplitz kernels for fast NUFFT operations, and splits k-t data into disjoint training and validation sets using configurable distributions (right-skewed, left-skewed, normal, uniform).

**`generator_320.py`** — Defines the `generatorNew` CNN module, a 10-layer convolutional architecture with nearest-neighbor upsampling that maps 30-dimensional latent vectors to complex-valued 320×320 image frames. Outputs real and imaginary channels separately, then recombines. Includes L1 weight regularization via `weightl1norm()`.

**`latentVariable.py`** — Defines the `latentVariableNew` class managing the temporal latent trajectory $l_{s,t} \in \mathbb{R}^{N_\text{frames} \times 30 \times 1 \times 1 \times N_\text{slices}}$. Implements KL divergence loss and temporal Tikhonov smoothness regularization.

**`optimize_gen_sub.py`** — Main training loop. Jointly optimizes generator parameters and latent vectors using Adam with a `ReduceLROnPlateau` scheduler driven by the held-out validation loss. Includes per-epoch validation, divergence detection, and checkpoint saving at minimum validation loss.

**`main_reconstruction.ipynb`** — Top-level script orchestrating the full pipeline per slice: data loading → 10-epoch initial training → 150-epoch final training → complex frame generation → saving 320×320 and 120×120 `.mat` outputs.

---

## ⚙️ Installation

### Requirements

- Python ≥ 3.8
- CUDA-capable GPU (≥ 24 GB VRAM recommended for 800 frames)

### Setup

```bash
git clone https://github.com/md-shahin-ali/self-supervised_reconstruction.git
cd self-supervised_reconstruction
pip install -r requirements.txt
```

### Key Dependencies

| Package | Purpose |
|---|---|
| `torch` | Neural network and GPU computation |
| `torchkbnufft` | Non-Cartesian k-space NUFFT and Toeplitz operators |
| `sigpy` | MRI coil sensitivity estimation (JSense) |
| `mat73` | Reading MATLAB v7.3 `.mat` files |
| `scipy` | Signal processing and generalized eigenvalue decomposition |
| `imageio` | GIF generation from magnitude frames |
| `scikit-learn` | PCA-based coil compression |
| `ismrmrdtools` | Inati / Walsh coil sensitivity methods |

Full list: see [`requirements.txt`](requirements.txt).

---

## 🚀 How to Run the Code

Ensure that the required dependencies are installed (see `requirements.txt`). After that, specify your data path and reconstruction parameters in `main_reconstruction.py` and run it accordingly. Feel free to adjust the reconstruction parameters, e.g., number of arms per frame, number of frames to reconstruct, latent vector size, and validation split ratio.
```bash
python main_reconstruction.py
```

This code builds upon the self-supervised variational manifold framework. Related conference abstracts from this line of work:

> 1. M. S. Ali et al., "Prospective validation of self-supervised spiral variational manifold learning for upper-airway collapse imaging," *ISMRM Workshop on Data Sampling and Image Reconstruction 2026*, Abstract 00243. [[Link]](https://echo.ismrm.org/abstracts/view/98d6f8bd-452e-4b21-9633-6c5434e4c661)
> 2. W. Alam et al., "Accelerated 3D dynamic upper-airway MRI in naturally sleeping obstructive sleep apnea patients," *ISMRM 2023*, Abstract 3078. [[Link]](https://archive.ismrm.org/2023/3078.html)
> 3. M. S. Ali et al., "Sensitivity analysis of self-supervised variational manifold learning based accelerated dynamic upper-airway collapse MRI," *ISMRM 2025*, Abstract 2595. [[Link]](https://archive.ismrm.org/2025/2595.html)

---

## Data Availability

The OSA patient data used in this study cannot be publicly released due to IRB and patient privacy restrictions. To request access for research collaboration, please contact the corresponding author.

---

## 📄 Citation

If you use this code or method in your research, please cite:

```bibtex
@article{ali2025prospective,
  title   = {Prospective validation of self-supervised spiral variational manifold
             learning for upper-airway collapse imaging},
  author  = {Ali, Md Shahin and Alam, Wahidul and Jacob, Mathews and
             Van Daele, Douglas and Liu, Junjie and Lingala, Sajan Goud},
  journal = {Magnetic Resonance in Medicine},
  year    = {2025},
  note    = {Under review}
}
```

---

## 🙏 Acknowledgements

This work was supported by the **National Institutes of Health** under grant **NIH NHLBI R01 HL173483**. MRI data were acquired on an instrument funded by NIH-S10 instrumentation grant **1S10OD025025-01**.

---

## 📬 Contact

This repository is intended to support reproducible research. If you encounter any issues or have questions about the code, feel free to open a GitHub issue or reach out directly at [mdshahin-ali@uiowa.edu](mailto:mdshahin-ali@uiowa.edu)
