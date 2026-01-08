import pickle
import numpy as np
import os
import sys

# --- PERCORSI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'forks', 'MotionAGFormer', 'data', 'motion3d')

OLD_PKL = os.path.join(DATA_DIR, 'h36m_sh_conf_cam_source_final.pkl')
NEW_PKL = os.path.join(DATA_DIR, 'h36m_sh_conf_cam_source_final_SPCI.pkl')

def compare():
    if not os.path.exists(NEW_PKL):
        print("ERRORE: Il file SPCI non esiste!")
        return

    print("Caricamento file ORIGINALE...")
    with open(OLD_PKL, 'rb') as f:
        data_old = pickle.load(f)

    print("Caricamento file SPCI...")
    with open(NEW_PKL, 'rb') as f:
        data_new = pickle.load(f)

    print("\n--- CONFRONTO CONFIDENZA (TRAIN SET) ---")
    
    # MotionAGFormer salva la confidenza in data['train']['confidence']
    # o concatena in 'joint_2d'. Lo script di iniezione ha creato la key 'confidence'.
    
    conf_old = None
    conf_new = None

    # Estrazione Old
    if 'confidence' in data_old['train']:
        conf_old = data_old['train']['confidence']
    else:
        # Se non c'è la key, era implicita (es. tutti 1 o nel joint_2d)
        print("Il file originale non ha la key 'confidence' esplicita.")

    # Estrazione New
    if 'confidence' in data_new['train']:
        conf_new = data_new['train']['confidence']
    else:
        print("ERRORE CRITICO: Il file SPCI non ha la key 'confidence'!")
        return

    if conf_old is not None:
        # Check uguaglianza
        is_same = np.array_equal(conf_old, conf_new)
        if is_same:
            print("\n!!! ALLARME: I dati di confidenza sono IDENTICI !!!")
            print("L'iniezione SPCI non ha funzionato o hai sovrascritto il file con quello vecchio.")
        else:
            print("\n>>> SUCCESSO: I dati di confidenza sono DIVERSI.")
            print(f"Media Confidenza Originale: {np.mean(conf_old):.4f}")
            print(f"Media Confidenza SPCI:      {np.mean(conf_new):.4f}")
            
            # Controllo inversione
            print(f"Min SPCI: {np.min(conf_new):.4f} (Dovrebbe essere > 0)")
            print(f"Max SPCI: {np.max(conf_new):.4f} (Dovrebbe essere <= 1)")
    else:
        print("\n>>> SUCCESSO PARZIALE: Il file nuovo ha una key 'confidence' che prima non esisteva.")
        print(f"Media Confidenza SPCI: {np.mean(conf_new):.4f}")

if __name__ == "__main__":
    compare()