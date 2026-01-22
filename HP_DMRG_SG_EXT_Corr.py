import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
import csv
import logging
import math
import time
import enum
import random

# --- TENPY IMPORTS ---
try:
    import tenpy
    from tenpy.networks.site import SpinHalfSite
    from tenpy.models.model import CouplingMPOModel
    from tenpy.models.lattice import Chain
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import dmrg
except ImportError:
    print("CRITICAL ERROR: 'tenpy' library not found. Please install using 'pip install physics-tenpy'.")
    sys.exit(1)

# Configure logging
logging.getLogger('tenpy').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
sys.setrecursionlimit(20000)

# =============================================================================
# PART 1: PATH UTILITIES
# =============================================================================

def load_path_from_csv(filename):
    path = []
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                path.append((int(row['x']), int(row['y'])))
        print(f"Successfully loaded {len(path)} steps from {filename}")
        return path
    except FileNotFoundError:
        print(f"ERROR: Could not find '{filename}'. Make sure to run the C++ optimizer first.")
        sys.exit(1)

def calculate_geometric_cost(path):
    node_to_idx = {node: i for i, node in enumerate(path)}
    total_cost = 0.0
    for i, u in enumerate(path):
        x, y = u
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for v in neighbors:
            if v in node_to_idx:
                idx_v = node_to_idx[v]
                dist = abs(i - idx_v)
                if dist > 1: 
                    total_cost += math.sqrt(dist)
    return total_cost * 0.5

# =============================================================================
# PART 2: DMRG ENGINE
# =============================================================================

class InlineDMRGEngine(dmrg.TwoSiteDMRGEngine):
    def sweep(self, *args, **kwargs):
        return super().sweep(*args, **kwargs)

class HeisenbergModel(CouplingMPOModel):
    def init_terms(self, model_params):
        bond_dict = model_params.get('bond_dict', {})
        xy_to_idx = model_params.get('xy_to_idx', {})
        
        for (u, v), strength in bond_dict.items():
            if u not in xy_to_idx or v not in xy_to_idx: continue
            
            i = xy_to_idx[u]
            j = xy_to_idx[v]
            if i > j: i, j = j, i
            
            self.add_coupling_term(strength, i, j, "Sz", "Sz")
            self.add_coupling_term(strength * 0.5, i, j, "Sp", "Sm")
            self.add_coupling_term(strength * 0.5, i, j, "Sm", "Sp")

def generate_bonds_dictionary(width, height):
    """
    Generates a dictionary of bonds with random couplings J = +1 or -1.
    """
    energies = {}
    random.seed(42)  # Fixed seed for reproducibility
    
    for x in range(width):
        for y in range(height):
            u = (x, y)
            # Horizontal Bond
            if x + 1 < width:
                v = (x + 1, y)
                J = random.choice([1.0, -1.0]) 
                energies[tuple(sorted((u, v)))] = J
            # Vertical Bond
            if y + 1 < height:
                v = (x, y + 1)
                J = random.choice([1.0, -1.0])
                energies[tuple(sorted((u, v)))] = J
    return energies

def define_model_from_path(L, bond_dict, path):
    site = SpinHalfSite(conserve='Sz')
    N_sites = len(path)
    xy_to_idx = {node: i for i, node in enumerate(path)}
    chain_lat = Chain(N_sites, site, bc='open')
    
    model_params = {
        'lattice': chain_lat,
        'bc_MPS': 'finite',
        'conserve': 'Sz',
        'bond_dict': bond_dict,
        'xy_to_idx': xy_to_idx
    }
    return HeisenbergModel(model_params)

def run_dmrg_schedule(model, chi_schedule):
    L_sites = model.lat.N_sites
    init_state = ["up", "down"] * (L_sites // 2)
    if L_sites % 2: init_state.append("up")
    
    psi = MPS.from_product_state(model.lat.mps_sites(), init_state, bc='finite')
    history = []
    
    for chi in chi_schedule:
        dmrg_params = {
            'mixer': True, 
            'max_E_err': 1.e-7, 
            'max_sweeps': 60, 
            'min_sweeps': 10,
            'max_trunc_err': 1.0, 
            'trunc_params': {'chi_max': chi, 'svd_min': 1.e-10},
            'verbose': 0
        }
        engine = InlineDMRGEngine(psi, model, dmrg_params)
        E, psi = engine.run()
        
        S_profile = psi.entanglement_entropy()
        max_S = np.max(S_profile)
        trunc_err = engine.sweep_stats.get('max_trunc_err', [0.0])[-1]
        history.append((chi, E, max_S, trunc_err))
        
    return history, psi

# =============================================================================
# PART 3: CORRELATION UTILITIES
# =============================================================================

def compute_correlation_pair(psi, i, j):
    """
    Computes <S_i . S_j> for two specific sites i and j (0-indexed).
    """
    C_zz = psi.correlation_function("Sz", "Sz", sites1=[i], sites2=[j])
    C_pm = psi.correlation_function("Sp", "Sm", sites1=[i], sites2=[j])
    C_mp = psi.correlation_function("Sm", "Sp", sites1=[i], sites2=[j])
    
    # Extract scalar from 1x1 matrix
    val = C_zz[0,0] + 0.5 * (C_pm[0,0] + C_mp[0,0])
    return val

def save_disorder_to_csv(L, bond_dict, filename):
    """
    Saves the spin glass J_ij realization.
    Rows are Horizontal bonds, then Vertical bonds.
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x1", "y1", "x2", "y2", "J"])
        for (u, v), J in bond_dict.items():
            writer.writerow([u[0], u[1], v[0], v[1], J])
    print(f"Saved disorder realization to {filename}")

def save_all_correlations_to_csv(psi, path, filename):
    """
    Computes all <S_i . S_j> pairs and saves to CSV.
    Includes both 1D path indices and 2D physical coordinates.
    """
    print("Computing full correlation matrix...")
    N_sites = psi.L
    
    # Compute full matrix
    C_zz = psi.correlation_function("Sz", "Sz")
    C_pm = psi.correlation_function("Sp", "Sm")
    C_mp = psi.correlation_function("Sm", "Sp")
    C_total = C_zz + 0.5 * (C_pm + C_mp)
    
    print(f"Saving {N_sites*(N_sites-1)//2} correlations to {filename}...")
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Added xi, yi, xj, yj columns
        writer.writerow(["idx_i", "idx_j", "xi", "yi", "xj", "yj", "correlation"])
        
        for i in range(N_sites):
            for j in range(i): 
                val = C_total[i, j]
                
                # LOOKUP: Map 1D index back to 2D coordinates
                xi, yi = path[i]
                xj, yj = path[j]
                
                writer.writerow([i, j, xi, yi, xj, yj, val])
                
    print("Done saving correlations.")

def save_individual_plots(L, results):
    metrics_config = [
        (1, r"$E_0$", f"Energy Convergence (L={L})", "Optimized_Energy.png", False),
        (2, r"$S_{vN}$", f"Entropy Scaling (L={L})", "Optimized_Entropy.png", False),
        (3, r"$\epsilon$", f"Error Scaling (L={L})", "Optimized_Error.png", True)
    ]
    
    for idx, ylabel, title, filename, is_log in metrics_config:
        plt.figure(figsize=(7, 5))
        for res in results:
            history = res['history']
            chis = [h[0] for h in history]
            vals = [h[idx] for h in history]
            plt.plot(chis, vals, 'o-', linewidth=2, label="Optimized Path", color='blue')
            
        plt.xlabel(r"Bond Dimension $\chi$", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        if is_log: plt.yscale('log')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    CSV_FILENAME = "paths.csv"
    CHI_SCHEDULE = [40, 80, 160] 
    TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
    
    print(f"--- DMRG Spin Glass Simulation (Optimized Path Only) ---")
    
    # 1. Load Path
    print(f"\n[1] Loading Optimized Path...")
    opt_path = load_path_from_csv(CSV_FILENAME)
    
    xs = [p[0] for p in opt_path]
    ys = [p[1] for p in opt_path]
    L = max(max(xs), max(ys)) + 1
    print(f"    Detected Lattice Size: L={L}")

    # 2. Define Physical System (Spin Glass)
    bond_dict = generate_bonds_dictionary(L, L)
    
    # Save Disorder Realization
    disorder_file = f"Disorder_Realization_{TIMESTAMP}.csv"
    save_disorder_to_csv(L, bond_dict, disorder_file)
    
    # 3. Run DMRG
    print(f"\n[2] Running DMRG Schedule...")
    geom_cost = calculate_geometric_cost(opt_path)
    print(f"    Geometric Cost: {geom_cost:.2f}")
    
    model = define_model_from_path(L, bond_dict, opt_path)
    history, psi = run_dmrg_schedule(model, CHI_SCHEDULE)
    
    final_E, final_S, final_Err = history[-1][1], history[-1][2], history[-1][3]
    print(f"    Final: E={final_E:.5f} | S={final_S:.4f} | Err={final_Err:.2e}")
    
    # 4. Save Correlations
    corr_file = f"Correlations_{TIMESTAMP}.csv"
    
    # --- FIX IS HERE: Pass 'opt_path', not 'L' ---
    save_all_correlations_to_csv(psi, opt_path, corr_file)
    
    # 5. Save Plots
    results = [{'label': 'Optimized', 'cost': geom_cost, 'history': history, 'color': 'blue'}]
    save_individual_plots(L, results)