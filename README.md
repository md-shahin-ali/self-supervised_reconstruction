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

$$C(\theta, l_{s,t}) = \sum_{s,t} \| \mathsf{E}_{\Gamma,s,t}(G_\theta(l_{s,t})) - \Omega(\Gamma) \|_2^2 + \sigma^2 R^{\text{scale}}_{l_{s,t}} + \lambda_1 R_{G_\theta} + \lambda_2 R^{\text{temporal Tikhonov}}_{l_{s,t}}$$

where $G_\theta$ is a CNN generator mapping latent vectors $l_{s,t}$ to multi-slice image volumes, $\Gamma$ is the training mask, and the terms enforce data consistency, variational latent prior, generator regularization, and temporal smoothness.

---

## Results

### Adaptive vs. Fixed Learning Rate

<p align="center">
  <img src="Figures/FIG4a.png" width="48%" alt="Fixed learning rate training"/>
  <img src="Figures/FIG4b.png" width="48%" alt="Adaptive learning rate training"/>
</p>

*Figure 2: Left: Fixed learning rate leads to overfitting and temporal artifacts at late epochs. Right: Validation-loss–guided adaptive learning rate stabilizes training and preserves temporal fidelity throughout optimization.*

### Effect of Validation-Set Splitting Distribution

<p align="center">
  <img src="Figures/FIG5.png" width="90%" alt="Effect of splitting distribution"/>
</p>

*Figure 3: Comparison of right-skewed, left-skewed, normal, and uniform splitting strategies. Right-skewed allocation of high-frequency samples to the training set achieves the best SER (22.78 dB) and sharpest air–tissue boundaries.*

### Prospective Reconstruction Comparison (OSA Patient, Natural Sleep)

<p align="center">
  <img src="Figures/FIG6.png" width="75%" alt="Reconstruction comparison"/>
</p>

*Figure 4: Mid-temporal spatiotemporal profiles from prospectively undersampled spiral data. The proposed self-supervised method preserves sharp air–tissue boundaries and smooth temporal evolution, outperforming parallel imaging, compressed sensing, view sharing, and analysis manifold.*

### Correlation with Physiological Signals

<p align="center">
  <img src="Figures/FIG7.png" width="90%" alt="Physiological signal correlation"/>
</p>

*Figure 5: Dynamic airway reconstruction from OSA0001 alongside respiratory effort (red) and SpO₂ (green). The model detects genuine collapse events — reflected in a 2–5% drop in oxygen saturation — that other methods fail to capture.*

### Temporal Airway Dynamics (12 Consecutive Frames)

<p align="center">
  <img src="Figures/FIG8.png" width="90%" alt="Temporal dynamics comparison"/>
</p>

*Figure 6: 12-frame comparison for OSA0007, Slice 6. The self-supervised model consistently resolves two distinct airway structures with smooth temporal evolution (frames 4–11), while competing methods show noise, blurring, or loss of the secondary airway structure.*

### Expert Image Quality Scores

<p align="center">
  <img src="Figures/FIG9.png" width="90%" alt="Expert scoring results"/>
</p>

*Figure 7: Expert image quality scores (1 = unacceptable, 2 = adequate, 3 = good, 4 = excellent) across aliasing, spatial blurring, and temporal blurring for 12 datasets. The proposed self-supervised method achieves the highest scores in all three categories (Kruskal–Wallis, p < 0.001).*

<div align="center">

| Method | Aliasing ↑ | Spatial Blurring ↑ | Temporal Blurring ↑ |
|:---:|:---:|:---:|:---:|
| **Self-Supervised (Ours)** | **3.92** | **4.00** | **4.00** |
| Analysis Manifold | 3.42 | 3.17 | 2.25 |
| View Sharing | 2.88 | 2.62 | 3.08 |
| Compressed Sensing | 1.75 | 2.04 | 2.29 |
| Parallel Imaging | 1.00 | 1.00 | 1.00 |

</div>

*Table 1: Kruskal–Wallis: aliasing H=98.57, spatial H=82.18, temporal H=76.69 (all p<0.001).*

### Reconstruction Artifacts

<p align="center">
  <img src="Figures/FIG10.png" width="90%" alt="Reconstruction artifacts"/>
</p>

*Figure 8: Artifacts (red circles) were observed in only 8 of 136 reconstructed slices (5.88%), appearing as spurious signal intensities or localized distortions. These are rare but motivate future work on anatomy-aware regularization.*

---

## Repository Structure

```
.
├── main_reconstruction.py       # Main reconstruction script (one-cell, slices 1–11)
├── dataOpNewKbnufft.py          # Data loading, k-space operators, coil sensitivity estimation
├── generator_320.py             # CNN generator (320×320 complex-valued output)
├── latentVariable.py            # Latent variable module with KL and smoothness regularization
├── optimize_gen_sub.py          # Training loop with self-supervised early stopping
├── Figures/                     # Paper figures (FIG1–FIG10)
├── Data/                        # (Not included) Raw .mat k-space data
├── requirements.txt             # Python dependencies
└── README.md
```

### Module Descriptions

**`dataOpNewKbnufft.py`** — Handles all data I/O and MRI forward/adjoint operators. Reads `.mat` or pickle k-space files, applies FOIVR-based virtual coil compression (PCA), estimates coil sensitivity maps via ESPIRiT / JSENSE / Inati / Walsh, precomputes Toeplitz kernels for fast NUFFT operations, and splits k-t data into disjoint training and validation sets using configurable distributions (right-skewed, left-skewed, normal, uniform).

**`generator_320.py`** — Defines the `generatorNew` CNN module, a 10-layer convolutional architecture with nearest-neighbor upsampling that maps 30-dimensional latent vectors to complex-valued 320×320 image frames. Outputs real and imaginary channels separately, then recombines. Includes L1 weight regularization via `weightl1norm()`.

**`latentVariable.py`** — Defines the `latentVariableNew` class managing the temporal latent trajectory $l_{s,t} \in \mathbb{R}^{N_\text{frames} \times 30 \times 1 \times 1 \times N_\text{slices}}$. Supports multiple initialization strategies (`ones`, `random`, `zeros`, `interpolate`). Implements KL divergence loss and temporal Tikhonov smoothness regularization.

**`optimize_gen_sub.py`** — Main training loop. Jointly optimizes generator parameters and latent vectors using Adam with a `ReduceLROnPlateau` scheduler driven by the held-out validation loss. Includes per-epoch validation, divergence detection, and checkpoint saving at minimum validation loss.

**`main_reconstruction.py`** — Top-level script orchestrating the full pipeline per slice: data loading → 10-epoch initial training → 150-epoch final training → complex frame generation → saving 320×320 and 120×120 `.mat` outputs → GIF creation from magnitude images.

---

## Installation

### Requirements

- Python ≥ 3.8
- CUDA-capable GPU (≥ 16 GB VRAM recommended for 11 slices)

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

## How to Run the Code

Ensure that the required dependencies are installed (see `requirements.txt`). After that, specify your data path and reconstruction parameters in `main_reconstruction.py` and run it accordingly. Feel free to adjust the reconstruction parameters, e.g., number of arms per frame, number of frames to reconstruct, latent vector size, and validation split ratio.

```bash
python main_reconstruction.py
```

This code builds upon the self-supervised variational manifold framework. A preliminary retrospective sensitivity analysis of this method was presented at ISMRM 2025:

> M. S. Ali et al., "Sensitivity analysis of self-supervised variational manifold learning based accelerated dynamic upper-airway collapse MRI," *ISMRM 2025*, Abstract 2595. [[Link]](https://archive.ismrm.org/2025/2595.html)

---

## Data Availability

The OSA patient data used in this study cannot be publicly released due to IRB and patient privacy restrictions. To request access for research collaboration, please contact the corresponding author.

---

## Citation

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

## Acknowledgements

This work was supported by the **National Institutes of Health** under grant **NIH NHLBI R01 HL173483**. MRI data were acquired on an instrument funded by NIH-S10 instrumentation grant **1S10OD025025-01**.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
