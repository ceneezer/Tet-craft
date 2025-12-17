import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# -----------------------------
# 1. Digital-root product on Z^2
# -----------------------------

def digital_root_product_array(I, J, base):
    """
    Vectorized digital root calculation following the explicit rule:
    - if i*j is 0, result is 0.
    - Otherwise, the result is |i*j| % (base-1).
    - If that remainder is 0, the final result is base-1 (e.g., 9).
    """
    p = I * J
    out = np.zeros_like(I, dtype=int)
    mask_nonzero = (p != 0)

    # --- UPDATED LOGIC TO BE MORE EXPLICIT ---
    # 1. Calculate the remainder for all non-zero products
    r = np.abs(p[mask_nonzero]) % (base - 1)

    # 2. Where the remainder is 0, set the value to base-1 (e.g., 9)
    r[r == 0] = base - 1

    # 3. Assign the final results to the output array
    out[mask_nonzero] = r

    return out


# -----------------------------
# 2. Build a finite grid of the pattern
# -----------------------------

def build_pattern_grid(base=10, periods_along=2, width_factor=1):
    """Build a finite integer grid (I,J) around the origin and compute the pattern."""
    period = base - 1
    half_len_i = period * periods_along
    I_vals = np.arange(-half_len_i, half_len_i + 1)
    half_len_j = max(1, period * width_factor // 2)
    J_vals = np.arange(-half_len_j, half_len_j + 1)
    I_grid, J_grid = np.meshgrid(I_vals, J_vals, indexing='ij')
    P = digital_root_product_array(I_grid, J_grid, base=base)
    return I_vals, J_vals, P


# -----------------------------
# 3. Map grid to a Möbius strip
# -----------------------------

def mobius_strip_coords(I_vals, J_vals, R=3.0, half_width=0.8):
    """Map the integer grid indices (i, j) to a Möbius strip in 3D."""
    I_min, I_max = I_vals[0], I_vals[-1]
    t_vals = 4.0 * np.pi * (I_vals - I_min) / (I_max - I_min)
    J_min, J_max = J_vals[0], J_vals[-1]
    v_vals = half_width * (2.0 * (J_vals - J_min) / (J_max - J_min) - 1.0)
    T, V = np.meshgrid(t_vals, v_vals, indexing='ij')
    X = (R + V * np.cos(T / 2.0)) * np.cos(T)
    Y = (R + V * np.cos(T / 2.0)) * np.sin(T)
    Z = V * np.sin(T / 2.0)
    return X, Y, Z


# -----------------------------
# 4. Create custom colormap
# -----------------------------

def create_custom_colormap(base=10):
    """Creates a colormap with a dark color for 0 and 'Reds' for values 1 to base-1."""
    sequential_cmap = plt.get_cmap('Reds', base - 1)
    colors = [sequential_cmap(i) for i in range(base - 1)]
    zero_color = (0.1, 0.1, 0.1, 1.0)
    final_colors = [zero_color] + colors
    cmap = ListedColormap(final_colors)
    bounds = np.arange(-0.5, base + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm

# -----------------------------
# 5. The Multi-View Visualization
# -----------------------------

def visualize_multi_view(base=10, periods_along=2, width_factor=2,
                         R=3.0, half_width=1.0):

    # --- 1. Data and Color Setup ---
    I_vals, J_vals, P = build_pattern_grid(base=base, periods_along=periods_along, width_factor=width_factor)
    X, Y, Z = mobius_strip_coords(I_vals, J_vals, R=R, half_width=half_width)
    cmap, norm = create_custom_colormap(base=base)

    j0_idx = np.where(J_vals == 0)[0][0]
    X0, Y0, Z0 = X[:, j0_idx], Y[:, j0_idx], Z[:, j0_idx]
    n_steps = len(X0)

    # --- 2. Figure and Subplot Layout ---
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"Digital-Root Pattern Explorer (Base {base})", fontsize=16)

    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax_fpv = fig.add_subplot(gs[1, 0])
    ax_flat = fig.add_subplot(gs[:, 1])

    # --- 3. Plot Static Content for Each View ---

    # View 1: 3D Möbius Strip (Top-Left)
    ax_3d.set_title("3D Möbius Strip View")
    ax_3d.plot_surface(X.T, Y.T, Z.T, rstride=1, cstride=1, facecolors=cmap(norm(P.T)),
                       linewidth=0, shade=False, zorder=1)
    ax_3d.plot(X0, Y0, Z0, color='cyan', linewidth=2, zorder=5)
    ax_3d.view_init(elev=30, azim=-50)
    ax_3d.set_axis_off()

    # View 2: 2D Flat Pattern Map (Right Side)
    ax_flat.set_title("2D Flat Pattern Map")
    ax_flat.imshow(P.T, cmap=cmap, norm=norm, origin='lower', aspect='auto',
                   extent=[I_vals.min(), I_vals.max(), J_vals.min(), J_vals.max()])
    ax_flat.set_xlabel("i coordinate (along strip length)")
    ax_flat.set_ylabel("j coordinate (across strip width)")
    ax_flat.set_xticks(np.arange(I_vals.min(), I_vals.max()+1, base-1))
    ax_flat.grid(True, linestyle='--', alpha=0.5)

    # View 3: First-Person View (Bottom-Left)
    ax_fpv.set_title("First-Person View")
    fpv_depth = 15
    fpv_width = len(J_vals)
    fpv_J_vals = J_vals

    fpv_image = ax_fpv.imshow(np.zeros((fpv_width, fpv_depth)), cmap=cmap, norm=norm, origin='lower',
                               extent=[-0.5, fpv_depth - 0.5, fpv_J_vals.min() - 0.5, fpv_J_vals.max() + 0.5])

    ax_fpv.plot(0, 0, '^', color='red', markersize=15, markeredgecolor='black',
                label='_You are here', zorder=20, clip_on=False)

    ax_fpv.set_xticks([])
    ax_fpv.set_xlabel("Direction of Travel")
    ax_fpv.set_yticks(J_vals[::2])
    ax_fpv.set_ylabel("Position Across Strip")
    ax_fpv.legend(loc='upper right')

    # --- 4. Initialize Animated Artists ---
    point_3d, = ax_3d.plot([], [], [], 'ro', markersize=8, zorder=10)
    point_2d, = ax_flat.plot([], [], 'ro', markersize=10)

    # --- 5. Animation Update Function ---
    def update(frame):
        idx = frame % n_steps

        point_3d.set_data([X0[idx]], [Y0[idx]])
        point_3d.set_3d_properties([Z0[idx]])

        point_2d.set_data([I_vals[idx]], [0])

        P_rolled = np.roll(P, shift=-idx, axis=0)
        data_slice = P_rolled[0:fpv_depth, :]
        fpv_image.set_data(data_slice.T)

        return point_3d, point_2d, fpv_image

    # --- 6. Run the Animation ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ani = FuncAnimation(fig, update, frames=2 * n_steps, interval=50, blit=False)
    plt.show()


if __name__ == "__main__":
    visualize_multi_view(base=10, periods_along=2, width_factor=2,
                         R=3.0, half_width=1.0)
