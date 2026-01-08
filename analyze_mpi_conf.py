import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'forks', 'MotionAGFormer', 'data', 'motion3d')
FILES_TO_INSPECT = ['data_train_3dhp.npz', 'data_test_3dhp.npz']

def inspect_deep():
    for filename in FILES_TO_INSPECT:
        file_path = os.path.join(DATA_DIR, filename)
        print(f"\n" + "="*60)
        print(f"ISPEZIONE PROFONDA: {filename}")
        print("="*60)
        
        if not os.path.exists(file_path):
            print("File non trovato.")
            continue
            
        try:
            with np.load(file_path, allow_pickle=True) as raw_wrapper:
                # 1. UNWRAP: Estraiamo il dizionario interno
                if 'data' not in raw_wrapper:
                    print(f"Chiave 'data' non trovata nel wrapper. Chiavi: {list(raw_wrapper.keys())}")
                    continue
                
                # Qui succede la magia: .item() recupera l'oggetto originale
                real_data = raw_wrapper['data'].item()
                
                print(f"Tipo contenuto estratto: {type(real_data)}")
                
                if not isinstance(real_data, dict):
                    print("Il contenuto non è un dizionario. Impossibile navigare.")
                    continue

                # 2. NAVIGAZIONE (MPI-INF spesso è diviso per Sequenze 'S1', 'S2'...)
                keys = list(real_data.keys())
                print(f"Chiavi nel dizionario: {keys[:10]} ... (Totale: {len(keys)})")
                
                # Cerchiamo dati in modo ricorsivo (max 1 livello di profondità per ora)
                found_conf = False
                
                # Iteriamo sulle prime chiavi per capire la struttura
                for k in keys[:5]: # Controlliamo solo i primi 5 per non intasare l'output
                    content = real_data[k]
                    print(f"\n--- Analisi Chiave '{k}' ---")
                    
                    # Caso A: Il dizionario contiene direttamente gli array (es. 'joint_2d')
                    if isinstance(content, np.ndarray):
                        print(f"  È un array: {content.shape}")
                        if len(content.shape) == 3 and content.shape[2] == 3:
                            plot_confidence(content[:, :, 2], f"{filename} - {k}")
                            found_conf = True
                            
                    # Caso B: Il dizionario contiene Sotto-Dizionari (es. 'joint_2d' è dentro)
                    elif isinstance(content, dict):
                        print(f"  È un dizionario con chiavi: {list(content.keys())}")
                        # Cerchiamo 'joint_2d' o simili qui dentro
                        for sub_k in content.keys():
                            if 'joint_2d' in sub_k or 'data_2d' in sub_k:
                                arr = content[sub_k]
                                print(f"    >>> TROVATO ARRAY '{sub_k}': {arr.shape}")
                                if arr.shape[-1] == 3:
                                    print("    >>> HA 3 CANALI! Estraggo confidenza...")
                                    plot_confidence(arr[:, :, 2], f"{filename} - {k}/{sub_k}")
                                    found_conf = True
                                elif arr.shape[-1] == 2:
                                    print("    >>> HA SOLO 2 CANALI (No Confidenza esplicita = 1.0).")
                
                if not found_conf:
                    print("\n>>> CONCLUSIONE: Non ho trovato array (N, 17, 3) espliciti nei campioni analizzati.")
                    print("    Molto probabilmente i dati sono (N, 17, 2) e la confidenza è implicita (1.0).")

        except Exception as e:
            print(f"Errore: {e}")

def plot_confidence(conf_data, title):
    flat_conf = conf_data.flatten()
    print(f"    Statistiche Confidenza:")
    print(f"      Min: {flat_conf.min():.4f}, Max: {flat_conf.max():.4f}, Mean: {flat_conf.mean():.4f}")
    
    unique = np.unique(flat_conf)
    if len(unique) < 10:
        print(f"      VALORI UNICI: {unique}")
        print("      >>> ALERT: Confidenza SINTETICA/BINARIA! <<<")
    
    plt.figure(figsize=(10, 3))
    plt.hist(flat_conf, bins=50, color='purple', alpha=0.7)
    plt.title(f"Confidence: {title}")
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    inspect_deep()