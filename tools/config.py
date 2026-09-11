# ---------- Codec ----------

NO_CODEBOOKS = 8

# ---------- Diffusion schedule ----------

# Total masked fraction at trajectory position u: R = (1 - u) / (1 - (1 - T_SHIFT) * u)
T_SHIFT = 0.1

# Maximum number of diffiusion steps utilized in inference.
MAX_STEPS = 16

# Maximum trajectory progress encountered during inference.
# Expressed directly by maximum number of steps.
U_MAX = (MAX_STEPS - 1) / MAX_STEPS

# ---------- MASKING ----------
# ---------- Per-codebook masking profile (adaptive) ----------

# rho_c = exp(-kappa * c), with kappa = KAPPA_0 + U(0, KAPPA_NOISE).
# See knowledge/mask_sampling_recipe.md.
KAPPA_0 = 1.033
KAPPA_NOISE = 0.02

# ---------- MASKING ----------
# ---------- Remasking correction (adaptive-remask) ----------

# Multipliers on rho_0 and rho_1, the two codebooks remasking touches.
# See knowledge/mask_sampling_remask.md.
PHI_0_BOUNDS = (0.10, 1.15)
PHI_1_BOUNDS = (0.30, 1.15)
