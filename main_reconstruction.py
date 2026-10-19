# ===============================================================
# ONE-CELL RECONSTRUCTION – SLICES 1–11 (or any range)
# → SAVES ONLY COMPLEX-VALUED RECONSTRUCTIONS (320×320 + cropped 120×120 complex)
# → 320×320 variable: Recon320
# → 120×120 variable: Recon120
# → GIF generated from 120×120 complex magnitude
# ===============================================================
import sys, os, importlib, pickle, tqdm
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import savemat
from matplotlib.transforms import Bbox
import imageio.v2 as imageio
# ------------------- SUPPRESS ALL PYTORCH WARNINGS -------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# ---------------------------------------------------------------------
# ---- FIX MEMORY FRAGMENTATION ---------------------------------------
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# ---------------------------------------------------------------------
# ---- add the folder that contains the 4 .py files -------------------
sys.path.append('/insert/your/directory/') # <-- change if needed
# ---- CORE IMPORTS ---------------------------------------------------
import dataOpNewKbnufft
from dataOpNewKbnufft import dataAndOperators
import generator_320
from generator_320 import generatorNew
import latentVariable
from latentVariable import latentVariableNew
import optimize_gen_sub
from optimize_gen_sub import optimize_generator
# ---- GPU ------------------------------------------------------------
gpu = torch.device('cuda:0')
plt.rcParams["figure.figsize"] = (15, 5)
# List to track successfully processed slices
processed_slices = []
# ---- GLOBAL PARAMS TEMPLATE -----------------------------------------
params_template = {
    'name': 'parameters',
    'directory': '',
    'device': gpu,
    'filename': "/Data/file_name.mat",
    'dtype': torch.float,
    'fastMode': True,
    'verbose': True,
    'im_size': (320, 320),
    'nintlPerFrame': 3,
    'nintlvsToDelete': 0,
    'nFramesDesired': 800,
    'factor': 1,
    'nBatch': 6,
    'gen_base_size': 60,
    'gen_reg': 0.000001,
    'virtual_coils': 8,
    'mask_size': 0.90,
    'coilEst': 'espirit',
    'siz_l': 30,
    'Undersample': False,
    'Usp_arms_PerFrame': 3,
    'splitRatio': 0.10,
    'ssTrainMode': True,
    'splitDist': 'rightSkewed',
    'CoilMapPrecomputed': False,
}
# ---- SIMPLE FIXED CENTER CROP FUNCTION ------------------------------
def central_crop(array, crop_size=(30, 30)):
    """
    Simple fixed center crop - same style as old code
    """
    h, w = array.shape[:2]
    start_h = max(0, (h - crop_size[0]) // 2)
    start_w = max(0, (w - crop_size[1]) // 2)
    end_h = start_h + crop_size[0]
    end_w = start_w + crop_size[1]
    return array[start_h:end_h, start_w:end_w]
# ===============================================================
# MAIN LOOP – processing slices (change range as needed)
# ===============================================================
for SLICE in range(1, 12):  # ← slices 1–11 (adjust as needed for all slices)
    print(f" DATA LOADED: {os.path.basename(params_template['filename'])}")
    print("\n" + "="*80)
    print(f" PROCESSING SLICE {SLICE}")
    print("="*80 + "\n")
    # -----------------------------------------------------------
    # 1. Per-slice params
    # -----------------------------------------------------------
    params = params_template.copy()
    params['slice'] = SLICE - 1 # 0-based index for data loading
    # -----------------------------------------------------------
    # 2. Load data
    # -----------------------------------------------------------
    importlib.reload(dataOpNewKbnufft)
    dop = dataAndOperators(params)
    print(f" -> Data loaded for slice {SLICE}")
    # -----------------------------------------------------------
    # 3. Generator + latent variable
    # -----------------------------------------------------------
    importlib.reload(generator_320)
    G = generatorNew(params).to(torch.float32).cuda(gpu)
    alpha = [0.1, 0.15, 0.2, 0.25, 0.3, 0.1, 0.9, 0.2, 0.1, 0.5] * 3
    alpha = torch.FloatTensor(alpha).to(gpu)
    importlib.reload(latentVariable)
    z = latentVariableNew(params, init='ones', alpha=alpha, klreg=1)
    # ------------------- INITIAL TRAINING -------------------
    print("\n" + "="*80)
    print(f" INITIAL TRAINING STARTED FOR SLICE {SLICE} (10 epochs)")
    print("="*80 + "\n")
    params['lr_g'] = 2e-4
    params['lr_z'] = 0.0
    train_epoch = 10
    importlib.reload(optimize_gen_sub)
    G, z, train_hist, epoch0 = optimize_generator(
        dop, G, z, params,
        train_epoch=train_epoch,
        save_weight=False,
        gtExists=False
    )
    print(f" -> Initial training done: epoch {epoch0}, G_loss = {train_hist['G_losses'][-1]:.6f}")
    # ---- save only the 10-epoch weights --------------------
    pathname = params['filename'].replace('.mat',
        f'_OnlyG_{params["gen_base_size"]}d/weights_onlyGenerator_{params["coilEst"]}')
    pathname = f'{pathname}_{train_epoch}_epoch{SLICE}_{params["nintlPerFrame"]}arms_{params["siz_l"]}latVec{params["nFramesDesired"]}frms'
    os.makedirs(pathname, exist_ok=True)
    init_path = os.path.join(pathname, f'init_weights_slice{SLICE}_epoch{train_epoch}.pth')
    torch.save({'G': G.state_dict(), 'z': z.z_}, init_path)
    print(f" -> Initial weights saved: {init_path}")
    # ------------------- FINAL TRAINING -------------------
    print("\n" + "="*80)
    print(f" FINAL TRAINING STARTED FOR SLICE {SLICE} (150 epochs)")
    print("="*80 + "\n")
    ckpt = torch.load(init_path, map_location=gpu, weights_only=True)
    importlib.reload(generator_320)
    G1 = generatorNew(params).to(torch.float32).cuda(gpu)
    G1.load_state_dict(ckpt['G'])
    importlib.reload(latentVariable)
    z1 = latentVariableNew(params, z_in=z, alpha=alpha, klreg=1)
    z1.z_ = ckpt['z']
    params['stop_training'] = 15
    params['lr_g'] = 2e-5
    params['lr_z'] = 4e-4
    final_epoch = 150
    importlib.reload(optimize_gen_sub)
    G1, z1, train_hist, epoch1 = optimize_generator(
        dop, G1, z1, params,
        train_epoch=final_epoch,
        save_weight=False,
        gtExists=False
    )
    print(f" -> Final training done: epoch {epoch1}, G_loss = {train_hist['G_losses'][-1]:.6f}")
    # ===============================================================
    # POST-TRAINING: Generate complex frames → save 320 complex → 120 complex → GIF
    # ===============================================================
    im_size_full   = (320, 320)
    im_size_crop   = (120, 120)
    n_frames       = params["nFramesDesired"]
    nsl            = 1
    TR             = 61e-3 / 11          # ≈ 5.545 ms per slice
    fps            = 1.0 / (params['nintlPerFrame'] * TR * nsl)
    dpi            = 500

    print(f" → Generating {n_frames} complex frames (full 320×320) for slice {SLICE} ...")

    images_full    = np.zeros((n_frames, *im_size_full, 2), dtype=np.float32)
    images_cropped = np.zeros((n_frames, *im_size_crop, 2), dtype=np.float32)

    for i in range(n_frames):
        with torch.no_grad():
            out = G1(z1.z_[i:i+1, :, :, :, 0])
            out = out.squeeze(0).detach().cpu()

            if out.ndim == 3 and out.shape[0] == 2:
                img_real = out[0].squeeze().numpy()
                img_imag = out[1].squeeze().numpy()
            elif out.is_complex():
                img_real = out.real.squeeze().numpy()
                img_imag = out.imag.squeeze().numpy()
            else:
                raise ValueError(f"Unexpected generator output shape/dtype: {out.shape}, is_complex={out.is_complex()}")

            images_full[i, ..., 0] = img_real
            images_full[i, ..., 1] = img_imag

            # Crop complex data (real & imag separately)
            crop_real = central_crop(img_real, crop_size=im_size_crop)
            crop_imag = central_crop(img_imag, crop_size=im_size_crop)
            images_cropped[i, ..., 0] = crop_real
            images_cropped[i, ..., 1] = crop_imag

    # ────────────────────────────────────────────────────────────────
    # Result folder (main directory)
    # ────────────────────────────────────────────────────────────────
    base = params['filename'].replace('.mat',
              f'_RECONS_{params["gen_base_size"]}d/results_{params["coilEst"]}')
    dirname = (f'{base}_{SLICE}thSlice_{params["nintlPerFrame"]}arms_'
               f'{params["splitRatio"]}SplitRatio_{params["siz_l"]}latVec'
               f'{params["nFramesDesired"]}frms_{train_epoch}_Geph{final_epoch}_feph')
    os.makedirs(dirname, exist_ok=True)

    # ────────────────────────────────────────────────────────────────
    # 1. First: Save full 320×320 complex
    # ────────────────────────────────────────────────────────────────
    uncropped_dir = os.path.join(dirname, "Uncropped")
    os.makedirs(uncropped_dir, exist_ok=True)

    mat_name_320 = f'Recon320_{SLICE}thSlice_{params["nintlPerFrame"]}arms_' \
                   f'{params["splitRatio"]}SplitRatio_{params["siz_l"]}latVec_' \
                   f'{n_frames}frms.mat'
    mat_path_320 = os.path.join(uncropped_dir, mat_name_320)

    savemat(mat_path_320, {
        "Recon320": images_full,
        "label": "Complex-valued reconstructions [frames × 320 × 320 × {real, imag}]"
    })

    print("\n" + "═" * 70)
    print("📦 320×320 COMPLEX RECONSTRUCTION SAVED")
    print("═" * 70)
    print(f"  • File:       {mat_name_320}")
    print(f"  • Location:   {uncropped_dir}")
    print(f"  • Full path:  {mat_path_320}")
    print(f"  • Variable:   Recon320")
    print(f"  • Shape:      {images_full.shape} (frames, height, width, 2)")
    print("═" * 70 + "\n")

    # ────────────────────────────────────────────────────────────────
    # 2. Then: Save cropped 120×120 complex (original file name style)
    # ────────────────────────────────────────────────────────────────
    mat_name_120 = f'Recon_{SLICE}thSlice_{params["nintlPerFrame"]}arms_' \
                   f'{params["splitRatio"]}SplitRatio_{params["siz_l"]}latVec_' \
                   f'{n_frames}frms.mat'
    mat_path_120 = os.path.join(dirname, mat_name_120)

    savemat(mat_path_120, {
        "Recon120": images_cropped,
        "label": "Complex-valued reconstructions [frames × 120 × 120 × {real, imag}]"
    })

    print("═" * 70)
    print("📦 120×120 COMPLEX RECONSTRUCTION SAVED")
    print("═" * 70)
    print(f"  • File:       {mat_name_120}")
    print(f"  • Location:   {dirname}")
    print(f"  • Full path:  {mat_path_120}")
    print(f"  • Variable:   Recon120")
    print(f"  • Shape:      {images_cropped.shape} (frames, 120, 120, 2)")
    print("═" * 70 + "\n")

    # ────────────────────────────────────────────────────────────────
    # 3. Last: GIF from 120×120 complex magnitude
    # ────────────────────────────────────────────────────────────────
    print("═" * 70)
    print("🎞️ MAGNITUDE GIF FROM 120×120 COMPLEX DATA")
    print("═" * 70)

    # Your best parameters
    GLOBAL_BRIGHTNESS = 1.5
    GLOBAL_GAMMA      = 1.0
    CLIP_LOW_PCT      = 1.0
    CLIP_HIGH_PCT     = 100.0

    magnitude = np.sqrt(images_cropped[..., 0]**2 + images_cropped[..., 1]**2)

    low  = np.percentile(magnitude, CLIP_LOW_PCT)
    high = np.percentile(magnitude, CLIP_HIGH_PCT)

    clipped = np.clip(magnitude, low, high)
    norm = (clipped - low) / (high - low + 1e-10)
    norm = norm * GLOBAL_BRIGHTNESS
    norm = np.clip(norm, 0.0, 1.0)

    if abs(GLOBAL_GAMMA - 1.0) > 1e-6:
        norm = np.power(norm, GLOBAL_GAMMA)

    norm = np.clip(norm, 0.0, 1.0)
    gif_frames_uint8 = (norm * 255).astype(np.uint8)

    gif_frames = []
    for k in range(n_frames):
        fig, ax = plt.subplots(1, figsize=(im_size_crop[1]/dpi, im_size_crop[0]/dpi), dpi=dpi)
        ax.set_position([0,0,1,1])
        ax.imshow(gif_frames_uint8[k], cmap='gray', vmin=0, vmax=255)
        ax.axis('off')

        png_path = f'{dirname}/frame_{k:04d}.png'
        fig.savefig(png_path,
                    bbox_inches=Bbox([[0,0],[im_size_crop[1]/dpi, im_size_crop[0]/dpi]]),
                    dpi=dpi, pad_inches=0)
        plt.close(fig)

        gif_frames.append(imageio.imread(png_path))
        os.remove(png_path)

    # GIF name matches the 120×120 complex .mat name
    gif_path = os.path.join(dirname, mat_name_120.replace('.mat', '.gif'))

    imageio.mimsave(gif_path, gif_frames, fps=fps, loop=0)

    print(f"  • File:       {os.path.basename(gif_path)}")
    print(f"  • Location:   {dirname}")
    print(f"  • Full path:  {gif_path}")
    print(f"  • Frames:     {n_frames}")
    print(f"  • FPS:        {fps:.4f}")
    print(f"  • TR/slice:   {TR:.6f} s")
    print("═" * 70 + "\n")

    # Clean memory
    del dop, G, z, G1, z1, images_full, images_cropped, gif_frames, ckpt
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    processed_slices.append(SLICE)

    print("\n" + "="*80)
    print(f" SLICE {SLICE} RECONSTRUCTION COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")

# ===============================================================
# FINAL SUMMARY
# ===============================================================
print("\n" + "="*80)
print("RECONSTRUCTION SUMMARY")
print("-"*80)
if processed_slices:
    for sl in sorted(processed_slices):
        print(f"  • Slice {sl} : completed")
    print("-"*80)
    print(f"Total: {len(processed_slices)} slice(s) successfully reconstructed")
else:
    print("No slices were processed.")
print("="*80)