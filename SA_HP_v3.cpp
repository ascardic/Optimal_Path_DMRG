#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <map>
#include <set>
#include <fstream>
#include <omp.h> // [Added] Required for OpenMP

// --- ENUMERATIONS ---
enum class Direction : uint8_t {
    NOWHERE = 0,
    NORTH = 1,
    EAST = 2,
    SOUTH = 3,
    WEST = 4
};

struct Point {
    int x, y;
    bool operator==(const Point& other) const { return x == other.x && y == other.y; }
    bool operator!=(const Point& other) const { return !(*this == other); }
    bool operator<(const Point& other) const { 
        if (x != other.x) return x < other.x;
        return y < other.y;
    }
};

// [Changed] Global RNG must be thread_local for parallel execution
// This ensures every CPU core gets its own unique random sequence.
thread_local std::mt19937 rng(std::random_device{}());

class Hamiltonian {
public:
    int width, height;
    Point start_node;
    std::vector<Direction> grid; // Flattened 2D array: index = y * width + x
    bool dirty;
    
    // Caches
    std::vector<Point> ordered_nodes;
    std::vector<int> path_indices; // Maps flattened index -> path index

    Hamiltonian(int w, int h) : width(w), height(h), dirty(true) {
        grid.resize(w * h, Direction::NOWHERE);
        path_indices.resize(w * h, -1);
        start_node = {0, 0};
        create_snake_path();
    }

    // --- GRID HELPERS ---
    inline int flat_idx(int x, int y) const { return y * width + x; }
    inline int flat_idx(Point p) const { return p.y * width + p.x; }
    
    inline Direction get_dir(Point p) const { 
        if (p.x < 0 || p.x >= width || p.y < 0 || p.y >= height) return Direction::NOWHERE;
        return grid[flat_idx(p)]; 
    }
    
    inline void set_dir(Point p, Direction d) { grid[flat_idx(p)] = d; }

    Direction zig_zag(int x, int y) {
        bool even = (y % 2 == 0);
        if ((x == 0 && even) || (x == width - 1 && !even)) return Direction::NORTH;
        return even ? Direction::WEST : Direction::EAST;
    }

    void create_snake_path() {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                grid[flat_idx(x, y)] = zig_zag(x, y);
            }
        }
        start_node = {0, 0};
        set_dir(start_node, Direction::NOWHERE);
        dirty = true;
    }

    // --- GILBERT CURVE ---
    int sgn(int x) { return (x < 0) ? -1 : (x > 0 ? 1 : 0); }

    Point gilbert_d2xy_r(int dst_idx, int cur_idx, int x, int y, int ax, int ay, int bx, int by) {
        int w = std::abs(ax + ay);
        int h = std::abs(bx + by);
        int dax = sgn(ax), day = sgn(ay);
        int dbx = sgn(bx), dby = sgn(by);
        int dx = dax + dbx, dy = day + dby;
        int di = dst_idx - cur_idx;

        if (h == 1) return {x + dax * di, y + day * di};
        if (w == 1) return {x + dbx * di, y + dby * di};

        int ax2 = ax / 2, ay2 = ay / 2;
        int bx2 = bx / 2, by2 = by / 2;
        int w2 = std::abs(ax2 + ay2);
        int h2 = std::abs(bx2 + by2);

        if (2 * w > 3 * h) {
            if ((w2 % 2) && (w > 2)) { ax2 += dax; ay2 += day; }
            int nxt_idx = cur_idx + std::abs((ax2 + ay2) * (bx + by));
            if (cur_idx <= dst_idx && dst_idx < nxt_idx)
                return gilbert_d2xy_r(dst_idx, cur_idx, x, y, ax2, ay2, bx, by);
            cur_idx = nxt_idx;
            return gilbert_d2xy_r(dst_idx, cur_idx, x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by);
        } else {
            if ((h2 % 2) && (h > 2)) { bx2 += dbx; by2 += dby; }
            int nxt_idx = cur_idx + std::abs((bx2 + by2) * (ax2 + ay2));
            if (cur_idx <= dst_idx && dst_idx < nxt_idx)
                return gilbert_d2xy_r(dst_idx, cur_idx, x, y, bx2, by2, ax2, ay2);
            cur_idx = nxt_idx;
            nxt_idx = cur_idx + std::abs((ax + ay) * ((bx - bx2) + (by - by2)));
            if (cur_idx <= dst_idx && dst_idx < nxt_idx)
                return gilbert_d2xy_r(dst_idx, cur_idx, x + bx2, y + by2, ax, ay, bx - bx2, by - by2);
            cur_idx = nxt_idx;
            return gilbert_d2xy_r(dst_idx, cur_idx, x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby),
                                  -bx2, -by2, -(ax - ax2), -(ay - ay2));
        }
    }

    void create_hilbert_path() {
        std::vector<Point> coords;
        int total_points = width * height;
        coords.reserve(total_points);

        int ax, ay, bx, by;
        if (width >= height) { ax=width; ay=0; bx=0; by=height; }
        else { ax=0; ay=width; bx=height; by=0; }

        for (int i = 0; i < total_points; ++i) {
            coords.push_back(gilbert_d2xy_r(i, 0, 0, 0, ax, ay, bx, by));
        }

        std::fill(grid.begin(), grid.end(), Direction::NOWHERE);
        start_node = coords[0];
        
        for (int i = 0; i < total_points; ++i) {
            Point curr = coords[i];
            if (i == 0) set_dir(curr, Direction::NOWHERE);
            
            if (i > 0) {
                Point prev = coords[i - 1];
                int dx = curr.x - prev.x;
                int dy = curr.y - prev.y;
                Direction d = Direction::NOWHERE;
                if (dx == 1) d = Direction::EAST;
                else if (dx == -1) d = Direction::WEST;
                else if (dy == 1) d = Direction::SOUTH;
                else if (dy == -1) d = Direction::NORTH;
                set_dir(prev, d);
            }
        }
        set_dir(coords.back(), Direction::NOWHERE);
        dirty = true;
    }

    // --- MUTATION ---
    Point move(Point p) {
        Direction d = get_dir(p);
        if (d == Direction::NOWHERE) return {-1, -1};
        int dx = 0, dy = 0;
        if (d == Direction::NORTH) dy = -1;
        else if (d == Direction::SOUTH) dy = 1;
        else if (d == Direction::EAST) dx = 1;
        else if (d == Direction::WEST) dx = -1;
        
        Point next_p = {p.x + dx, p.y + dy};
        if (next_p.x < 0 || next_p.x >= width || next_p.y < 0 || next_p.y >= height) return {-1, -1};
        return next_p;
    }

    bool set_loop(Point start, Point stop, std::vector<Point>& loop_storage) {
        loop_storage.clear();
        Point p = start;
        int count = 0;
        int limit = width * height + 1;
        while (p.x != -1 && count < limit && p != stop && get_dir(p) != Direction::NOWHERE) {
            p = move(p);
            if (p.x != -1) loop_storage.push_back(p);
            count++;
        }
        return (p == stop);
    }

    void modify_path(Point pt_a, Point pt_b) {
        Direction pta = get_dir(pt_a);
        Direction ptb = get_dir(pt_b);
        Direction orientation = pta;
        
        if (orientation == Direction::NORTH || orientation == Direction::SOUTH) {
            if (pt_a.x < pt_b.x) { pta = Direction::EAST; ptb = Direction::WEST; }
            else { pta = Direction::WEST; ptb = Direction::EAST; }
        } else {
            if (pt_a.y < pt_b.y) { pta = Direction::SOUTH; ptb = Direction::NORTH; }
            else { pta = Direction::NORTH; ptb = Direction::SOUTH; }
        }
        set_dir(pt_a, pta);
        set_dir(pt_b, ptb);
    }
    
    bool split_grid(std::pair<Point, Point>& result, std::vector<Point>& temp_loop) {
        std::vector<std::pair<Point, Point>> candidates;
        candidates.reserve(width * height); 

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Point pt = {x, y};
                Direction dx = get_dir(pt);
                Point cx = {-1, -1};
                Direction req_dir = Direction::NOWHERE;

                if (dx == Direction::NORTH) { cx = {x + 1, y - 1}; req_dir = Direction::SOUTH; }
                else if (dx == Direction::SOUTH) { cx = {x + 1, y + 1}; req_dir = Direction::NORTH; }
                else if (dx == Direction::EAST) { cx = {x + 1, y + 1}; req_dir = Direction::WEST; }
                else if (dx == Direction::WEST) { cx = {x - 1, y + 1}; req_dir = Direction::EAST; }

                if (cx.x >= 0 && cx.x < width && cx.y >= 0 && cx.y < height) {
                    if (get_dir(cx) == req_dir) {
                        candidates.push_back({pt, cx});
                    }
                }
            }
        }

        if (candidates.empty()) return false;
        
        std::uniform_int_distribution<> dist(0, candidates.size() - 1);
        std::pair<Point, Point> choice = candidates[dist(rng)];
        
        if (set_loop(choice.first, choice.second, temp_loop)) {
            result = choice;
            return true;
        } else if (set_loop(choice.second, choice.first, temp_loop)) {
            result = {choice.second, choice.first};
            return true;
        }
        return false;
    }

    bool mend_grid(std::pair<Point, Point> sp, std::vector<Point>& curr_loop, std::pair<Point, Point>& result) {
        std::vector<std::pair<Point, Point>> candidates;
        
        std::vector<bool> in_loop(width * height, false);
        for(auto& p : curr_loop) in_loop[flat_idx(p)] = true;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Point pt = {x, y};
                bool lx = in_loop[flat_idx(pt)];
                Direction dx = get_dir(pt);
                Point cx = {-1, -1};
                Direction req_dir = Direction::NOWHERE;

                if (dx == Direction::NORTH) { cx = {x + 1, y - 1}; req_dir = Direction::SOUTH; }
                else if (dx == Direction::SOUTH) { cx = {x + 1, y + 1}; req_dir = Direction::NORTH; }
                else if (dx == Direction::EAST) { cx = {x + 1, y + 1}; req_dir = Direction::WEST; }
                else if (dx == Direction::WEST) { cx = {x - 1, y + 1}; req_dir = Direction::EAST; }

                if (cx.x != -1 && cx.x < width && cx.y < height && cx.x >=0 && cx.y >=0) {
                     if (get_dir(cx) == req_dir) {
                         bool rx = in_loop[flat_idx(cx)];
                         if (rx != lx) {
                             candidates.push_back({pt, cx});
                         }
                     }
                }
            }
        }

        auto it = std::remove(candidates.begin(), candidates.end(), sp);
        if (it != candidates.end()) candidates.erase(it, candidates.end());
        
        std::pair<Point, Point> sp_rev = {sp.second, sp.first};
        it = std::remove(candidates.begin(), candidates.end(), sp_rev);
        if (it != candidates.end()) candidates.erase(it, candidates.end());

        if (candidates.empty()) {
            result = sp;
            return false;
        }
        
        std::uniform_int_distribution<> dist(0, candidates.size() - 1);
        result = candidates[dist(rng)];
        return true;
    }

    bool mutate_step() {
        std::vector<Point> temp_loop;
        std::pair<Point, Point> sp;
        if (!split_grid(sp, temp_loop)) return false;
        
        modify_path(sp.first, sp.second);
        
        std::pair<Point, Point> tu;
        mend_grid(sp, temp_loop, tu);
        
        modify_path(tu.first, tu.second);
        dirty = true;
        return true;
    }

    // --- COST CALCULATION ---
    void build_index_cache() {
        std::vector<bool> is_target(width * height, false);
        for(int i=0; i<width*height; ++i) {
            if (grid[i] != Direction::NOWHERE) {
                int x = i % width; 
                int y = i / width;
                Direction d = grid[i];
                int nx=x, ny=y;
                if(d==Direction::NORTH) ny--;
                if(d==Direction::SOUTH) ny++;
                if(d==Direction::EAST) nx++;
                if(d==Direction::WEST) nx--;
                if(nx>=0 && nx<width && ny>=0 && ny<height) 
                    is_target[ny*width + nx] = true;
            }
        }
        
        int start_idx = 0;
        for(int i=0; i<width*height; ++i) {
            if(!is_target[i]) { start_idx = i; break; }
        }

        Point curr = {start_idx % width, start_idx / width};
        ordered_nodes.clear();
        std::fill(path_indices.begin(), path_indices.end(), -1);
        
        int idx = 0;
        while(true) {
            path_indices[flat_idx(curr)] = idx;
            ordered_nodes.push_back(curr);
            
            Direction d = get_dir(curr);
            if (d == Direction::NOWHERE) break;
            
            int dx = 0, dy = 0;
            if (d == Direction::NORTH) dy = -1;
            else if (d == Direction::SOUTH) dy = 1;
            else if (d == Direction::EAST) dx = 1;
            else if (d == Direction::WEST) dx = -1;
            
            Point next_p = {curr.x + dx, curr.y + dy};
            if (flat_idx(next_p) >= (int)path_indices.size() || path_indices[flat_idx(next_p)] != -1) break; 
            curr = next_p;
            idx++;
        }
        dirty = false;
    }

    double calculate_cost() {
        if (dirty) build_index_cache();
        double total_cost = 0.0;
        
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                int idx_u = path_indices[y * width + x];
                if (idx_u == -1) continue;
                
                Point neighbors[4] = {{x+1, y}, {x-1, y}, {x, y+1}, {x, y-1}};
                for (auto& n : neighbors) {
                    if (n.x >= 0 && n.x < width && n.y >= 0 && n.y < height) {
                        int idx_v = path_indices[n.y * width + n.x];
                        if (idx_v != -1) {
                            int dist = std::abs(idx_u - idx_v);
                            if (dist > 1) total_cost += std::sqrt(dist);
                        }
                    }
                }
            }
        }
        return total_cost * 0.5;
    }
};

double run_simulated_annealing(Hamiltonian& h, int max_iterations, double initial_temp, double cooling_rate) {
    double current_cost = h.calculate_cost();
    double best_cost = current_cost;
    std::vector<Direction> best_grid = h.grid;
    double T = initial_temp;

    for (int i = 0; i < max_iterations; ++i) {
        std::vector<Direction> prev_grid = h.grid;
        if (!h.mutate_step()) continue;

        double new_cost = h.calculate_cost();
        double delta = new_cost - current_cost;

        bool accepted = false;
        if (delta < 0) accepted = true;
        else {
            std::uniform_real_distribution<> dis(0.0, 1.0);
            if (std::exp(-delta / T) > dis(rng)) accepted = true;
        }

        if (accepted) {
            current_cost = new_cost;
            if (new_cost < best_cost) {
                best_cost = new_cost;
                best_grid = h.grid;
            }
        } else {
            h.grid = prev_grid; // Revert
            h.dirty = true;
        }

        T *= cooling_rate;
        if (T < 0.0001) break;
    }
    h.grid = best_grid;
    h.dirty = true;
    return best_cost;
}

int main() {
    int side = 10;
    int num_restarts = 100;
    int max_iter = 150000;
    double start_temp = 34.0;
    double cooling = 0.99995;

    std::cout << "Optimizing Hamiltonian Path (" << side << "x" << side << ") on M2..." << std::endl;
    std::cout << "Parallel execution enabled with OpenMP." << std::endl;

    double global_best_cost = 1e9;
    std::vector<Direction> global_best_grid; // Store the winning layout
    
    auto start_time = std::chrono::high_resolution_clock::now();

    // [Added] PARALLEL REGION STARTS HERE
    // 'dynamic' schedule helps balance load if some runs finish quicker than others
    #pragma omp parallel for schedule(dynamic)
    for (int run = 0; run < num_restarts; ++run) {
        Hamiltonian h_curr(side, side);
        h_curr.create_hilbert_path();
        
        double cost = run_simulated_annealing(h_curr, max_iter, start_temp, cooling);

        // [Added] CRITICAL REGION
        // Only one thread at a time can execute this block to print/update safely
        #pragma omp critical
        {
            std::cout << "Run " << (run + 1) << " finished. Cost: " << std::fixed << std::setprecision(1) << cost << std::endl;

            if (cost < global_best_cost) {
                global_best_cost = cost;
                global_best_grid = h_curr.grid; // Save valid grid
                std::cout << "-> New Global Best Found!" << std::endl;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_time - start_time;

    std::cout << "\nGlobal Best Cost: " << global_best_cost << std::endl;
    std::cout << "Total Time: " << total_duration.count() << "s" << std::endl;

    // --- WRITE OUTPUT FILE ---
    Hamiltonian h_final(side, side);
    h_final.grid = global_best_grid;
    h_final.dirty = true;
    h_final.build_index_cache(); // Fills h_final.ordered_nodes

    std::cout << "Writing paths.csv..." << std::endl;
    std::ofstream outFile("paths.csv");
        
    if (outFile.is_open()) {
        outFile << "step,x,y\n"; // CSV Header
        for (size_t i = 0; i < h_final.ordered_nodes.size(); ++i) {
            outFile << i << "," 
                    << h_final.ordered_nodes[i].x << "," 
                    << h_final.ordered_nodes[i].y << "\n";
        }
        outFile.close();
        std::cout << "Done." << std::endl;
    } else {
        std::cerr << "Error opening paths.csv for writing." << std::endl;
    }

    return 0;
}