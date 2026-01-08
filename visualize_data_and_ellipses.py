import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'forks', 'MotionAGFormer', 'data', 'motion3d')

PKL_OLD = os.path.join(DATA_DIR, 'h36m_sh_conf_cam_source_final.pkl')
PKL_NEW = os.path.join(DATA_DIR, 'h36m_sh_conf_cam_source_final_SPCI.pkl')

# Connessioni per disegnare lo scheletro (solo per riferimento visivo)
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), 
    (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)
]

def load_pkl(path):
    print(f"Caricamento {os.path.basename(path)}...")
    with open(path, 'rb') as f:
        return pickle.load(f)

def analyze_and_plot():
    if not os.path.exists(PKL_NEW):
        print("Errore: File SPCI non trovato.")
        return

    data_old = load_pkl(PKL_OLD)
    data_new = load_pkl(PKL_NEW)
    
    # Dati
    joints = data_new['train']['joint_2d'] # (N, 17, 2)
    conf_new_raw = data_new['train']['confidence'][:, 0, 0] # SPCI score
    
    # Recupero Raggio Reale in Pixel
    # Conf = 1 / (1 + R)  =>  R = (1/Conf) - 1
    radii_spci = (1.0 / (conf_new_raw + 1e-9)) - 1.0

    # Confidenza Vecchia (SH)
    if 'confidence' in data_old['train']:
        conf_old = data_old['train']['confidence'][:, 0, 0]
    else:
        conf_old = np.ones(len(joints))

    # --- 1. PLOT ISTOGRAMMI CON SCALA UNIFICATA ---
    print("\nGenerazione Istogrammi Comparativi...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Scala fissa 0-1 per confronto diretto
    axes[0].hist(conf_old, bins=50, range=(0, 1), color='gray', alpha=0.7, label='SH (Old)')
    axes[0].set_title("Original SH Confidence")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Confidence Score (0-1)")
    axes[0].legend()

    axes[1].hist(conf_new_raw, bins=50, range=(0, 1), color='blue', alpha=0.7, label='SPCI (Current)')
    axes[1].set_title("SPCI Confidence (Current)")
    axes[1].set_xlim(0, 1) # Qui si vedrà che sono tutti schiacciati a sinistra!
    axes[1].set_xlabel("Confidence Score (0-1)")
    axes[1].legend()

    # Raggi in pixel
    axes[2].hist(radii_spci, bins=50, range=(0, 30), color='orange', alpha=0.7)
    axes[2].axvline(x=np.mean(radii_spci), color='red', linestyle='--', label=f'Avg: {np.mean(radii_spci):.1f}px')
    axes[2].set_title("Real Uncertainty Radius (Pixels)")
    axes[2].set_xlabel("Radius (px)")
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

    # --- 2. VISUALIZZAZIONE SCHELETRO + GROUND TRUTH ---
    print("\nCercando un frame con movimento interessante...")
    
    # Cerchiamo un frame dove il raggio è > 10px (movimento/incertezza)
    # E dove c'è un movimento effettivo tra t e t+1
    diff = np.linalg.norm(joints[1:] - joints[:-1], axis=2).mean(axis=1) # Movimento medio
    
    # Filtra frame con Raggio > 8px e Movimento > 2px
    candidates = np.where((radii_spci[:-1] > 8) & (diff > 2))[0]
    
    if len(candidates) > 0:
        idx = np.random.choice(candidates)
    else:
        idx = np.random.randint(0, len(joints)-1)
        
    current_pose = joints[idx]       # Frame t (Input)
    next_pose = joints[idx+1]        # Frame t+1 (Ground Truth del movimento)
    radius = radii_spci[idx]
    
    print(f"Visualizzando Frame {idx}")
    print(f"Raggio SPCI: {radius:.2f} px")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Disegna Ossa (Frame t)
    for p1, p2 in SKELETON_EDGES:
        x = [current_pose[p1, 0], current_pose[p2, 0]]
        y = [current_pose[p1, 1], current_pose[p2, 1]]
        ax.plot(x, y, 'k-', linewidth=1, alpha=0.3)

    # Disegna Giunture
    covered_count = 0
    
    for j in range(17):
        cx, cy = current_pose[j] # Centro predizione (Frame t)
        tx, ty = next_pose[j]    # Verità (Frame t+1)
        
        # 1. Cerchio Incertezza
        circle = patches.Circle((cx, cy), radius=radius, 
                                edgecolor='orange', facecolor='orange', alpha=0.2, zorder=1)
        ax.add_patch(circle)
        
        # 2. Posizione Attuale (Blu)
        ax.scatter(cx, cy, c='blue', s=30, label='Frame t' if j==0 else "", zorder=2)
        
        # 3. Posizione Futura/GT (Verde X)
        ax.scatter(tx, ty, c='green', marker='x', s=40, label='Frame t+1 (GT)' if j==0 else "", zorder=3)
        
        # Check Copertura
        dist = np.sqrt((tx-cx)**2 + (ty-cy)**2)
        if dist <= radius:
            covered_count += 1
        else:
            # Linea rossa se fuori
            ax.plot([cx, tx], [cy, ty], 'r-', linewidth=1, alpha=0.5, zorder=1)

    # Info Testuali
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(f"Frame {idx} | SPCI Radius: {radius:.2f}px | Coverage: {covered_count}/17 joints")
    ax.legend()
    
    # Zoom su una parte interessante se lo scheletro è grande
    # (Opzionale: commenta se vuoi vedere tutto)
    # min_x, max_x = np.min(current_pose[:,0]), np.max(current_pose[:,0])
    # min_y, max_y = np.min(current_pose[:,1]), np.max(current_pose[:,1])
    # ax.set_xlim(min_x - 50, max_x + 50)
    # ax.set_ylim(max_y + 50, min_y - 50) # Y invertita
    
    plt.show()

if __name__ == "__main__":
    analyze_and_plot()