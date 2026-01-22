import matplotlib.pyplot as plt
import csv
import sys

# --- Configuration ---
DEFAULT_FILENAME = 'paths.csv'

# --- 1. Data Reading ---
def read_data(filename):
    """
    Reads the CSV output from the C++ simulation.
    Returns 1-based coordinates (adding 1 to raw 0-based C++ output).
    """
    x, y = [], []
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x.append(int(row['x']) + 1)
                y.append(int(row['y']) + 1)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        exit()
    return x, y

# --- 2. Cost Calculation ---
def calculate_cost(x_vals, y_vals):
    """
    Calculates cost: sum of sqrt(distance) for spatial neighbors 
    that are not sequential in the path.
    """
    path_indices = { (x, y): i for i, (x, y) in enumerate(zip(x_vals, y_vals)) }
    total_cost = 0.0
    
    for (x, y), idx_x in path_indices.items():
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for nx, ny in neighbors:
            if (nx, ny) in path_indices:
                idx_y = path_indices[(nx, ny)]
                dist = abs(idx_x - idx_y)
                if dist > 1:
                    total_cost += dist**(0.5)
    return total_cost * 0.5

# --- 3. Hilbert Path Generation (Benchmark) ---
class HilbertGenerator:
    """
    Generates a Generalized Hilbert Curve (Gilbert Curve).
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def sgn(self, x):
        return -1 if x < 0 else (1 if x > 0 else 0)

    def gilbert_d2xy_r(self, dst_idx, cur_idx, x, y, ax, ay, bx, by):
        w = abs(ax + ay)
        h = abs(bx + by)
        (dax, day) = (self.sgn(ax), self.sgn(ay))
        (dbx, dby) = (self.sgn(bx), self.sgn(by))
        
        di = dst_idx - cur_idx
        if h == 1: return (x + dax*di, y + day*di)
        if w == 1: return (x + dbx*di, y + dby*di)
        
        (ax2, ay2) = (ax//2, ay//2)
        (bx2, by2) = (bx//2, by//2)
        w2 = abs(ax2 + ay2)
        h2 = abs(bx2 + by2)
        
        if 2*w > 3*h:
            if (w2 % 2) and (w > 2): (ax2, ay2) = (ax2 + dax, ay2 + day)
            nxt_idx = cur_idx + abs((ax2 + ay2)*(bx + by))
            if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
                return self.gilbert_d2xy_r(dst_idx, cur_idx, x, y, ax2, ay2, bx, by)
            cur_idx = nxt_idx
            return self.gilbert_d2xy_r(dst_idx, cur_idx, x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by)
        
        if (h2 % 2) and (h > 2): (bx2, by2) = (bx2 + dbx, by2 + dby)
        nxt_idx = cur_idx + abs((bx2 + by2)*(ax2 + ay2))
        if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
            return self.gilbert_d2xy_r(dst_idx, cur_idx, x, y, bx2, by2, ax2, ay2)
        cur_idx = nxt_idx
        nxt_idx = cur_idx + abs((ax + ay)*((bx - bx2) + (by - by2)))
        if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
            return self.gilbert_d2xy_r(dst_idx, cur_idx, x+bx2, y+by2, ax, ay, bx-bx2, by-by2)
        
        cur_idx = nxt_idx
        return self.gilbert_d2xy_r(dst_idx, cur_idx, x+(ax-dax)+(bx2-dbx), y+(ay-day)+(by2-dby), -bx2, -by2, -(ax-ax2), -(ay-ay2))

    def generate(self):
        coords = []
        w, h = self.width, self.height
        total_points = w * h
        if w >= h: args = (0, 0, 0, w, 0, 0, h)
        else:      args = (0, 0, 0, 0, h, w, 0)

        for idx in range(total_points):
            coords.append(self.gilbert_d2xy_r(idx, *args))
            
        # Unpack and shift to 1-based indexing
        x_vals = [c[0] + 1 for c in coords]
        y_vals = [c[1] + 1 for c in coords]
        return x_vals, y_vals

# --- 4. Plotting ---
def setup_axis(ax, x_vals, y_vals, title, side):
    """Helper to apply common styling to an axis."""
    min_x, max_x = 1, side
    min_y, max_y = 1, side
    
    # Background Grid (dots)
    all_x = [x for x in range(min_x, max_x + 1) for _ in range(min_y, max_y + 1)]
    all_y = [y for _ in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]
    ax.scatter(all_x, all_y, color='lightgray', s=20, zorder=0)

    # Path
    ax.plot(x_vals, y_vals, marker='o', markersize=4, linewidth=2, color='teal', label='Path')

    # Markers
    ax.plot(x_vals[0], y_vals[0], marker='o', markersize=10, color='green', label='Start')
    ax.plot(x_vals[-1], y_vals[-1], marker='o', markersize=10, color='red', label='End')

    # Ticks and Labels
    if side > 16:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xticks(range(min_x, max_x + 1))
        ax.set_yticks(range(min_y, max_y + 1))
        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=12)

    # Limits
    ax.set_xlim(min_x - 0.5, max_x + 0.5)
    ax.set_ylim(min_y - 0.5, max_y + 0.5)
    
    # Title and Legend with Larger Fonts
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    ax.legend(loc='lower left', fontsize=12)

def plot_comparison(hilbert_data, optim_data, side):
    h_x, h_y = hilbert_data
    o_x, o_y = optim_data
    
    h_cost = calculate_cost(h_x, h_y)
    o_cost = calculate_cost(o_x, o_y)
    
    # 2-Panel Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    setup_axis(axes[0], h_x, h_y, f"Benchmark: Generalized Hilbert Path\n(Cost: {h_cost:.1f})", side)
    setup_axis(axes[1], o_x, o_y, f"Optimized Path (C++ Output)\n(Cost: {o_cost:.1f})", side)
    
    output_filename = f"Best_Path{side}.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Comparison figure saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILENAME
    print(f"Reading data from {input_file}...")
    
    # 1. Read Optimized Data
    o_x, o_y = read_data(input_file)
    
    # 2. Determine Side Length
    if not o_x:
        print("Empty data.")
        exit()
    side = max(max(o_x), max(o_y))
    
    # 3. Generate Hilbert Benchmark
    print(f"Generating Hilbert path for L={side}...")
    hg = HilbertGenerator(side, side)
    h_x, h_y = hg.generate()
    
    # 4. Plot
    plot_comparison((h_x, h_y), (o_x, o_y), side)