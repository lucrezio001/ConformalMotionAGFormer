import pickle
import numpy as np
import os
import sys

# --- CONFIGURA IL PERCORSO QUI ---
# Usa os.path per sicurezza, adattalo se necessario
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, 'forks', 'MotionAGFormer', 'data', 'motion3d', 'h36m_sh_conf_cam_source_final.pkl')

def inspect_data():
    if not os.path.exists(PKL_PATH):
        print(f"ERRORE: File non trovato in: {PKL_PATH}")
        return

    print(f"Caricamento {PKL_PATH}...")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)

    print("\n--- STRUTTURA GENERALE ---")
    print(f"Keys principali: {list(data.keys())}")

    for split in ['train', 'test']:
        if split not in data:
            continue
            
        print(f"\n--- {split.upper()} SET ---")
        split_data = data[split]
        print(f"Keys nel {split}: {list(split_data.keys())}")
        
        # Ispezioniamo joint_2d
        if 'joint_2d' in split_data:
            joints = split_data['joint_2d']
            print(f"Tipo di joint_2d: {type(joints)}")
            if isinstance(joints, np.ndarray):
                print(f"Shape di joint_2d: {joints.shape}")
            elif isinstance(joints, list):
                print(f"joint_2d è una LISTA di lunghezza {len(joints)}")
                print(f"Shape primo elemento: {np.array(joints[0]).shape}")
        
        # Ispezioniamo source (fondamentale per raggruppare i video)
        if 'source' in split_data:
            sources = split_data['source']
            print(f"Tipo di source: {type(sources)}")
            print(f"Lunghezza source: {len(sources)}")
            print(f"Esempio primi 5 source: {sources[:5]}")
            
            # Controllo coerenza
            if len(sources) == len(joints):
                print(">> OK: Numero di frame corrisponde al numero di etichette source.")
            else:
                print(">> ATTENZIONE: Discrepanza tra frame e source!")

if __name__ == "__main__":
    inspect_data()