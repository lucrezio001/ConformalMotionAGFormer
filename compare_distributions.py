import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Il file generato dalla Random Forest (o Ridge)
PKL_PATH = os.path.join(BASE_DIR, 'forks', 'MotionAGFormer', 'data', 'motion3d', 'SCPI_experiment', 'ridge_regression', 'h36m_sh_conf_cam_source_final_SPCI.pkl')

def compare_normalization():
    if not os.path.exists(PKL_PATH):
        print("File PKL non trovato!")
        return

    print(f"Caricamento {os.path.basename(PKL_PATH)}...")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)

    # Estraiamo i dati grezzi salvati (che sono 1/(1+W))
    # Prendiamo solo il TRAIN per calcolare le statistiche, ma plottiamo la distribuzione
    current_conf = data['train']['confidence'][:, 0, 0] # (N,)

    # 1. RECUPERO WIDTH GREZZA (Pixel)
    # Formula inversa: W = (1/C) - 1
    raw_width = (1.0 / (current_conf + 1e-9)) - 1.0
    raw_width = np.maximum(raw_width, 0) # Clip a 0 per pulizia numerica
    
    # Rimuoviamo outlier estremi per le statistiche (99° percentile)
    limit_99 = np.percentile(raw_width, 99)
    print(f"Statistiche Width (Pixel):")
    print(f"  Min: {np.min(raw_width):.2f}")
    print(f"  Max (assoluto): {np.max(raw_width):.2f}")
    print(f"  99° Percentile (usato per scale): {limit_99:.2f}")
    print(f"  Media: {np.mean(raw_width):.2f}")

    # --- STRATEGIA A: Min-Max Scaling (Inverso) ---
    # Vogliamo: Width 0 -> Conf 1.0 | Width Max -> Conf 0.0
    # Usiamo il 99° percentile come "Max" per non schiacciare tutto a causa di un outlier
    w_min = 0 # Sappiamo che il minimo teorico è 0
    w_max = limit_99
    
    # Clip dei dati al range
    w_clipped = np.clip(raw_width, w_min, w_max)
    # Normalizzazione 0-1 invertita
    conf_minmax = 1.0 - ((w_clipped - w_min) / (w_max - w_min))

    # --- STRATEGIA B: Standardizzazione (Z-Score) ---
    # Formula: Z = (x - mu) / sigma
    # Attenzione: Questo produce valori negativi e positivi (es. -2, +2)
    # Non è ideale per "Confidenza 0-1", ma vediamo come si distribuisce.
    mu = np.mean(raw_width)
    sigma = np.std(raw_width)
    conf_zscore = (raw_width - mu) / sigma
    
    # --- PLOTTING ---
    plt.figure(figsize=(20, 6))

    # 1. Raw Width (La realtà fisica)
    plt.subplot(1, 4, 1)
    plt.hist(raw_width, bins=100, range=(0, limit_99 * 1.2), color='orange', alpha=0.7)
    plt.title("1. Raw Uncertainty Width (Pixels)")
    plt.xlabel("Pixel Error Radius")
    plt.ylabel("Count")
    plt.axvline(x=np.mean(raw_width), color='red', linestyle='--', label='Mean')
    plt.legend()

    # 2. Current Formula (Iperbolica)
    plt.subplot(1, 4, 2)
    plt.hist(current_conf, bins=100, range=(0, 1), color='gray', alpha=0.7)
    plt.title("2. Current Formula: 1 / (1+W)")
    plt.xlabel("Confidence Score")
    plt.yticks([]) # Nascondo asse Y per pulizia

    # 3. Min-Max Normalized (Lineare)
    plt.subplot(1, 4, 3)
    plt.hist(conf_minmax, bins=100, range=(0, 1), color='blue', alpha=0.7)
    plt.title("3. Min-Max Normalized (Linear 0-1)")
    plt.xlabel("Confidence Score (High=Good)")
    plt.yticks([])

    # 4. Standardization (Z-Score)
    plt.subplot(1, 4, 4)
    plt.hist(conf_zscore, bins=100, range=(-2, 5), color='green', alpha=0.7)
    plt.title("4. Standardization (Z-Score)")
    plt.xlabel("Sigma from Mean (Low=Good)")
    plt.axvline(x=0, color='red', linestyle='--')
    plt.yticks([])

    plt.tight_layout()
    plt.show() # O plt.savefig('comparison.png') se preferisci

if __name__ == "__main__":
    compare_normalization()