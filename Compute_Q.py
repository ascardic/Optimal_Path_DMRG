import csv
import glob
import os
import sys
import math

def get_latest_correlation_file():
    """Finds the most recent Correlations_*.csv file in the current directory."""
    files = glob.glob("Correlations_*.csv")
    if not files:
        return None
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def compute_edwards_anderson_Q(filename):
    print(f"Loading data from: {filename}")
    
    correlations = []
    max_coord = 0
    
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = float(row['correlation'])
                correlations.append(val)
                
                # distinct coordinates to find L
                xi = int(row['xi'])
                yi = int(row['yi'])
                xj = int(row['xj'])
                yj = int(row['yj'])
                max_coord = max(max_coord, xi, yi, xj, yj)
                
    except FileNotFoundError:
        print("Error: File not found.")
        sys.exit(1)

    # 1. Determine Lattice Size L and Total Sites N
    # Coordinates are 0-indexed, so L is max_coord + 1
    L = max_coord + 1
    N = L * L
    print(f"Detected Lattice: L={L} (N={N} sites)")

    # 2. Calculate Sum of Squared Correlations
    # The file contains only unique off-diagonal pairs (i > j).
    sum_sq_off_diag = sum(c**2 for c in correlations)
    
    # We must account for:
    # A. Symmetry: C_ij^2 == C_ji^2 (multiply off-diagonal sum by 2)
    # B. Diagonal: C_ii^2. For Spin-1/2, <S_i . S_i> = 3/4.
    
    C_ii = 0.75
    sum_sq_diag = N * (C_ii**2)
    
    total_sum_sq = sum_sq_diag + (2 * sum_sq_off_diag)
    
    # 3. Compute Q
    # Formula: Q = (1 / L^4) * Sum_{i,j} C_ij^2
    # Note: L^4 = N^2
    normalization = 1.0 / (L**4)
    Q = total_sum_sq * normalization
    
    print("-" * 30)
    print(f"Sum Squares (Off-Diag): {sum_sq_off_diag:.6f}")
    print(f"Sum Squares (Diagonal): {sum_sq_diag:.6f}")
    print(f"Total Sum Squares     : {total_sum_sq:.6f}")
    print("-" * 30)
    print(f"Edwards-Anderson Q    : {Q:.6f}")
    print("-" * 30)

if __name__ == "__main__":
    # Check if user provided a specific file arg, otherwise find newest
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = get_latest_correlation_file()
        
    if target_file:
        compute_edwards_anderson_Q(target_file)
    else:
        print("No 'Correlations_*.csv' files found in the current directory.")