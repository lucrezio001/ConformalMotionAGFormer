import pickle
import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
spci_path = os.path.join(BASE_DIR, 'forks', 'MultiDimSPCI')
sys.path.append(spci_path)

try:
    from helpers.MultiDim_SPCI_class import SPCI_and_EnbPI
except ImportError as e:
    print(f"Errore importazione SPCI da {spci_path}")
    raise e

# --- INPUT E OUTPUT ---
# File originale
PATH_TO_PKL = os.path.join(BASE_DIR, 'forks', 'MotionAGFormer', 'data', 'motion3d', 'h36m_sh_conf_cam_source_final.pkl')

# 1. Output per MotionAGFormer (Pronto all'uso)
OUTPUT_PKL = os.path.join(BASE_DIR, 'forks', 'MotionAGFormer', 'data', 'motion3d', 'h36m_sh_conf_cam_source_final_SPCI.pkl')

# 2. Output per Debug (Analisi future)
DEBUG_DIR = os.path.join(BASE_DIR, 'debug_data')
os.makedirs(DEBUG_DIR, exist_ok=True)
OUTPUT_DEBUG_PKL = os.path.join(DEBUG_DIR, 'spci_matrices_intervals.pkl')
OUTPUT_PLOTS_DIR = os.path.join(DEBUG_DIR, 'plots')
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

# --- PARAMETRI ---
SEED = 104           
N_SPLITS = 5        
BOOTSTRAP_NUM = 15  
ALPHA = 0.01         

# Seed Globale
np.random.seed(SEED)

def get_video_intervals(source_list):
    intervals = []
    if len(source_list) == 0: return intervals
    current_src = source_list[0]
    start_idx = 0
    for i in range(1, len(source_list)):
        if source_list[i] != current_src:
            intervals.append((start_idx, i, current_src))
            current_src = source_list[i]
            start_idx = i
    intervals.append((start_idx, len(source_list), current_src))
    return intervals

def prepare_timeseries_data(joints_array):
    data_flat = joints_array.reshape(joints_array.shape[0], -1)
    X = data_flat[:-1] # t-1
    Y = data_flat[1:]  # t
    return X, Y

def plot_video_segment(Y_true, Y_pred_base, width, video_name, joint_idx=0, coord_idx=0):
    feat_idx = joint_idx * 2 + coord_idx
    T = len(Y_true)
    steps = np.arange(T)
    
    plt.figure(figsize=(15, 5))
    
    # Traiettoria Reale
    plt.plot(steps, Y_true[:, feat_idx], 'k-', label='Ground Truth', linewidth=1)
    
    # Predizione Base
    plt.plot(steps, Y_pred_base[:, feat_idx], 'b--', label='Base Pred', alpha=0.5)
    
    # Confidenza SPCI
    lower_bound = Y_pred_base[:, feat_idx] - width
    upper_bound = Y_pred_base[:, feat_idx] + width
    plt.fill_between(steps, lower_bound, upper_bound, color='orange', alpha=0.3, label=f'SPCI Interval (Alpha={ALPHA})')
    
    # Punti fuori
    out_of_bounds = (Y_true[:, feat_idx] < lower_bound) | (Y_true[:, feat_idx] > upper_bound)
    if np.any(out_of_bounds):
        plt.scatter(steps[out_of_bounds], Y_true[out_of_bounds, feat_idx], c='red', s=10, label='Out of Interval', zorder=5)
    
    plt.title(f"{video_name} | Joint {joint_idx} Coord {coord_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, f"{video_name}_j{joint_idx}.png"))
    plt.close()

def process_video_segment(video_data, video_name, do_plot=False):
    X_full, Y_full = prepare_timeseries_data(video_data)
    
    # Se troppo corto, skip
    if len(Y_full) < 20:
        return None

    # Container risultati
    uncertainty_radii = np.zeros(len(Y_full))
    Y_pred_all = np.zeros_like(Y_full)
    matrices_list = []
    
    # Statistiche Coverage
    coverage_hits = 0
    total_points = 0
    
    real_splits = min(N_SPLITS, len(Y_full) // 10)
    if real_splits < 2: return None

    kf = KFold(n_splits=real_splits, shuffle=False)
    
    for train_index, test_index in kf.split(X_full):
        X_train, X_test = X_full[train_index], X_full[test_index]
        Y_train, Y_test = Y_full[train_index], Y_full[test_index]
        
        #model = Ridge(alpha=1.0)
        model = RandomForestRegressor(n_estimators=10, max_depth=10, n_jobs=6, random_state=SEED)
        spci = SPCI_and_EnbPI(X_train, X_test, Y_train, Y_test, model)
        
        # Fit
        spci.fit_bootstrap_models_online_multistep(B=BOOTSTRAP_NUM, stride=1)
        spci.compute_Widths_Ensemble_online(alpha=ALPHA, stride=1, smallT=True, use_SPCI=False)
        
        # Salvataggio Matrice (Globale per il fold)
        matrices_list.append(spci.global_cov)
        
        # Ricostruzione predizioni base per debug/plot
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        Y_pred_all[test_index] = preds

        # Widths
        widths = spci.Width_Ensemble
        radii = widths['upper'].values - widths['lower'].values
        uncertainty_radii[test_index] = radii
        
        # Calcolo Coverage (Semplificato: media elementi dentro)
        diff = np.abs(Y_test - preds)
        hits = np.all(diff <= radii[:, None], axis=1) # Congiunta
        coverage_hits += np.sum(hits)
        total_points += len(hits)

        del spci

    # Risultati finali video
    mean_coverage = coverage_hits / total_points if total_points > 0 else 0
    
    # Plotting del primo video
    if do_plot:
        # Plot Joint 0 (Bacino)
        plot_video_segment(Y_full, Y_pred_all, uncertainty_radii, video_name, joint_idx=0, coord_idx=0)
        # Plot Joint 15 (Piede/Caviglia - Mobile)
        plot_video_segment(Y_full, Y_pred_all, uncertainty_radii, video_name, joint_idx=15, coord_idx=0)

    # Pad del primo frame mancante (copia del secondo)
    full_uncertainty = np.concatenate(([uncertainty_radii[0]], uncertainty_radii))

    return {
        'matrices': matrices_list,
        'uncertainty_radii': full_uncertainty, # Array (T,)
        'coverage': mean_coverage,
        'mean_width': np.mean(full_uncertainty)
    }

def main():
    if not os.path.exists(PATH_TO_PKL):
        print(f"ERROR: File not found at {PATH_TO_PKL}")
        return

    print(f"Loading dataset from {PATH_TO_PKL}...")
    with open(PATH_TO_PKL, 'rb') as f:
        data = pickle.load(f)
        
    debug_storage = {} 

    # PROCESSA SIA TRAIN CHE TEST
    for key in ['train', 'test']:
        print(f"\n--- Processing {key.upper()} set ---")
        
        joint_2d_all = data[key]['joint_2d']
        source_list = data[key]['source']
        
        video_intervals = get_video_intervals(source_list)
        print(f"Found {len(video_intervals)} videos.")
        
        # Array gigante per la CONFIDENZA da salvare nel PKL finale
        total_frames = len(source_list)
        new_confidence_array = np.zeros((total_frames, 17, 1), dtype=np.float32)
        
        # Metriche per il report
        key_coverages = []
        key_widths = []

        for i, (start, end, vid_name) in enumerate(tqdm(video_intervals)):
            video_slice = joint_2d_all[start:end]
            
            # Plotta solo il primo video di ogni set
            do_plot = (i == 0)
            
            try:
                # ESECUZIONE SPCI
                res = process_video_segment(video_slice, vid_name, do_plot=do_plot)
                
                if res is not None:
                    # 1. LOGICA DEBUG
                    key_coverages.append(res['coverage'])
                    key_widths.append(res['mean_width'])
                    
                    # Salviamo i dati debug solo per un sottoinsieme o tutti?
                    # Salviamo tutto, la dimensione è gestibile.
                    debug_storage[vid_name] = {
                        'cov_matrices': res['matrices'],
                        'intervals_radii': res['uncertainty_radii'],
                        'coverage': res['coverage']
                    }
                    
                    # 2. LOGICA PRODUCTION (Iniezione Confidenza)
                    # Recupera l'incertezza grezza (Raggio)
                    raw_uncertainty = res['uncertainty_radii']
                    
                    # INVERSIONE: Confidenza = 1 / (1 + Incertezza)
                    # Alto Raggio -> Bassa Confidenza
                    confidence_score = 1.0 / (1.0 + raw_uncertainty)
                    
                else:
                    # Fallback per video cortissimi
                    confidence_score = np.ones(len(video_slice))
                
                # Espansione a (T, 17, 1)
                conf_expanded = np.repeat(confidence_score[:, np.newaxis], 17, axis=1)
                conf_expanded = conf_expanded[:, :, np.newaxis]
                
                # Inserimento nell'array globale
                new_confidence_array[start:end] = conf_expanded
                
            except Exception as e:
                print(f"Error on {vid_name}: {e}")
                # Fallback sicuro
                fallback = np.ones((len(video_slice), 17, 1))
                new_confidence_array[start:end] = fallback

        # AGGIORNAMENTO DIZIONARIO PRINCIPALE
        print(f"Updating '{key}' confidence...")
        data[key]['confidence'] = new_confidence_array
        
        # REPORT PARZIALE
        print(f"REPORT {key.upper()}: Avg Coverage: {np.mean(key_coverages):.4f} | Avg Width: {np.mean(key_widths):.4f}")

    # SALVATAGGI FINALI
    print(f"\nSalvataggio Debug Data in {OUTPUT_DEBUG_PKL}...")
    with open(OUTPUT_DEBUG_PKL, 'wb') as f:
        pickle.dump(debug_storage, f)
        
    print(f"Salvataggio DATASET FINALE per MotionAGFormer in {OUTPUT_PKL}...")
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(data, f)
        
    print("\nDONE! Tutto pronto.")

if __name__ == "__main__":
    main()