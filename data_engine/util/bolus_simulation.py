import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

class GammaVariate:
    """
    Gamma-variate function for contrast concentration profile.
    I(t) = A * (t - t0)^alpha * exp(-(t - t0) / beta)
    A is implicitly handled by normalization later, here we just compute the shape.
    """

    def __init__(self, t0=0.0, alpha=2.0, beta=1.0):
        self.t0 = t0
        self.alpha = alpha
        self.beta = beta

    def sample(self, t):
        t_eff = t - self.t0
        # Only compute for t >= t0 (contrast arrives)
        if t_eff < 0:
            return 0.0
        # Use a small epsilon to prevent (t-t0)^alpha from becoming 0 when t=t0
        t_eff = max(t_eff, 1e-6)
        
        # Gamma-variate shape
        concentration = (t_eff ** self.alpha) * np.exp(-t_eff / self.beta)
        return concentration

# Renaming the old Gauss class for completeness, although it's no longer used
class Gauss_old: 
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma
    def sample(self, x):
        return np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)


class VesselNode:
    """The Class for a vessel node. d is the diameter."""

    def __init__(self, x=0.0, y=0.0, z=0.0, d=0.0, next=[]):
        self.x = x
        self.y = y
        self.z = z
        self.d = d
        self.next = next


def coordinates2vessel(coords, interpolate=0):
    """(Unchanged logic for creating vessel tree)"""
    
    def strand2vessel(strand):
        # Start with first node and connect each node
        head = VesselNode()
        node = head
        i = 0
        while i < strand.shape[1]:
            nextN = VesselNode(
                x=strand[0, i], y=strand[1, i], z=strand[2, i], d=strand[3, i]
            )
            node.next = [nextN]
            node = node.next[0]
            i += 1
        # We reached the end of this strand. See if there are any strands beginning
        # at this coordinate. If so, call them recursively to form the complete
        # vessel tree.
        child_strands = [
            s for j, s in enumerate(strands) if np.isclose(s[:, 0], strand[:, -1]).all()
        ]
        node.next = [strand2vessel(s) for s in child_strands]
        return head.next[0]
    
    # This function is used inside
    def interpolate_strands(strands, factor=2):
        strands = strands.copy()
        for s in range(len(strands)):
            strand = strands[s]
            for _ in range(0, factor, 2):
                strand_interp = np.zeros((strand.shape[0], strand.shape[1] * 2 - 1))
                for i in range(strand.shape[1]):
                    strand_interp[:, i * 2] = strand[:, i]
                    if i + 1 < strand.shape[1]:
                        interp = (strand[:, i] + strand[:, i + 1]) / 2
                        strand_interp[:, i * 2 + 1] = interp

                strand = strand_interp
            strands[s] = strand
        return strands
    
    # Get individual strands. Generates a list of coordinate chunks separated by
    # nan-rows in the original coordinate array.
    strands = []
    i = j = 0
    while i < coords.shape[1]:
        if j >= coords.shape[1] or np.isnan(coords[0, j]):
            strands.append(coords[:, i:j])
            i = j + 1
        j += 1

    # Interpolate with factor interpolate=0,2,4,...
    if interpolate:
        strands = interpolate_strands(strands, factor=interpolate)

    # Generate vessel tree
    return strand2vessel(strands[0])


def compute_contrast_dynamics(head, add_gamma=False, V_base=1.0, P_flow=1.0, alpha=2.0, beta=1.0,
                              branch_velocity_jitter=0.0, dispersion_gain=0.0, rng=None):
    """Assign arrival times and gamma-variate curves to every node of the tree.

    Arrival time: ``t0(child) = t0(parent) + L / v`` with the segment velocity
    ``v = V_base * d_child ** P_flow * j`` (Murray's-law-consistent flow, ``j`` a
    per-branch log-normal jitter, see below).

    Diversity controls (both default to 0 = the original behaviour):

    * ``branch_velocity_jitter``: standard deviation of ``log j``.  Each child
      segment draws its own factor, so sibling branches fill at different speeds
      and the tree is never filled perfectly symmetrically (as in real
      angiograms, where daughter branches of similar calibre opacify at
      different times).
    * ``dispersion_gain``: the bolus disperses along its path; the time constant
      of a node's gamma-variate is ``beta * (1 + dispersion_gain * t0 / max_t0)``
      so distal segments have a broader, lower and later-peaking enhancement
      curve than the trunk.  With 0 every node has the same curve shape.
    """
    rng = np.random.default_rng() if rng is None else rng

    # Pass 1: arrival times (depth-first).
    head.t0 = 0.0
    max_t0 = 0.0
    stack = [head]
    while stack:
        node = stack.pop()
        max_t0 = max(max_t0, node.t0)
        for child in node.next:
            segment_length = ((node.x - child.x) ** 2 + (node.y - child.y) ** 2 + (node.z - child.z) ** 2) ** 0.5
            jitter = float(np.exp(rng.normal(0.0, branch_velocity_jitter))) if branch_velocity_jitter > 0 else 1.0
            segment_velocity = V_base * (child.d ** P_flow) * jitter
            time_to_traverse = segment_length / segment_velocity if segment_velocity > 1e-6 else float("inf")
            child.t0 = node.t0 + time_to_traverse
            stack.append(child)

    # Pass 2: enhancement curves (need max_t0 for the dispersion law).
    if add_gamma:
        stack = [head]
        while stack:
            node = stack.pop()
            spread = 1.0 + dispersion_gain * (node.t0 / max_t0 if max_t0 > 0 else 0.0)
            node.gamma = GammaVariate(t0=node.t0, alpha=alpha, beta=beta * spread)
            stack.extend(node.next)

    return head, max_t0


def plot_vessel(head, max_dist=None, time_step=None, title=""):
    """(Modified to use t0 and GammaVariate)"""
    # Setup plot and colorbar
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    cax = fig.add_axes([0.15, 0.25, 0.02, 0.5])
    ax.view_init(azim=-139, elev=-145)
    sm = cm.ScalarMappable(cmap=cm.Blues)
    fig.colorbar(sm, cax=cax)
    cax.yaxis.set_ticks_position("left")

    # Plot the vessel
    stack = [head]
    while stack:
        head = stack.pop()
        if head:
            for child in head.next:
                if time_step != None and hasattr(head, 'gamma'):
                    # Plot bolus injection given a time step (using Gamma-variate)
                    # We normalize concentration by its max possible value (at peak)
                    # For Gamma: Max value is at t_peak = t0 + alpha*beta
                    
                    t_peak = head.gamma.t0 + head.gamma.alpha * head.gamma.beta
                    max_conc = head.gamma.sample(t_peak)
                    
                    conc = head.gamma.sample(time_step)
                    
                    # Normalized concentration (Intensity I)
                    c = conc / (max_conc if max_conc > 1e-6 else 1.0) 
                    
                    ax.plot(
                        [head.x, child.x],
                        [head.y, child.y],
                        [head.z, child.z],
                        linewidth=0.5 * head.d,
                        c=cm.Blues(c),
                    )
                elif hasattr(head, 'dist'):
                    # Plot distances (original logic)
                    ax.plot(
                        [head.x, child.x],
                        [head.y, child.y],
                        [head.z, child.z],
                        linewidth=0.5 * head.d,
                        c=cm.Blues(head.dist / max_dist),
                    )
                else:
                    # Plot Vessel tree in uniform color
                    ax.plot(
                        [head.x, child.x],
                        [head.y, child.y],
                        [head.z, child.z],
                        linewidth=0.5 * head.d,
                        color="Blue",
                    )
                stack.append(child)
    return fig


def print_vessel(head):
    """(Modified to print t0 instead of dist)"""
    stack = [head]
    while stack:
        node = stack.pop()
        if node:
            print(
                f"x: {node.x}, y:{node.y}, z:{node.z}, d:{node.d}, t0:{node.t0 if hasattr(node, 't0') else 'N/A'}"
            )
            stack.extend(node.next)


def coordinates_back(head, t):
    """Write the tree back as a ``5 x N`` array ``[x, y, z, d, I]`` for the voxeliser.

    ``process_network_fluid`` rasterises a segment between every pair of
    *consecutive non-NaN* columns, so the branch structure has to be encoded in
    the column order: every parent->child edge is written as its own two-point
    segment followed by a NaN separator column.  (An earlier version returned the
    plain depth-first node list without separators, which made the voxeliser draw
    spurious straight segments between the end of one branch and the start of the
    next, so the rendered trees looked like disconnected crossing rods.)

    Intensities are the gamma-variate concentration at time ``t`` normalised by
    the highest peak concentration in the tree.
    """
    # Highest peak concentration over the tree (normalisation constant).
    max_overall_conc = 1e-6
    stack = [head]
    while stack:
        node = stack.pop()
        if hasattr(node, "gamma"):
            t_peak = node.gamma.t0 + node.gamma.alpha * node.gamma.beta
            max_overall_conc = max(max_overall_conc, node.gamma.sample(t_peak))
        stack.extend(node.next)

    def column(node):
        intensity = node.gamma.sample(t) / max_overall_conc if hasattr(node, "gamma") else 0.0
        return [node.x, node.y, node.z, node.d, intensity]

    separator = [np.nan] * 5
    columns = []
    stack = [head]
    while stack:
        node = stack.pop()
        for child in node.next:
            columns.append(column(node))
            columns.append(column(child))
            columns.append(separator)
            stack.append(child)
    if not columns:  # a single isolated node
        columns = [column(head)]
    return np.array(columns, dtype=float).T


def bolus_injection(coords, t, alpha=2.0, beta=1.0, V_base=1.0, P_flow=1.0, interp_coords_factor=0,
                    branch_velocity_jitter=0.0, dispersion_gain=0.0, rng=None, head=None):
    """Input
        [[x1, y1, z1, d1],
         ...
        ]
    Return
          [[x1, y1, z1, d1, I1],
         ...
        ]
    t is the time step. alpha and beta are Gamma-variate parameters.
    V_base and P_flow define the segment velocity based on diameter: V_s = V_base * d^P_flow.
    Returns ``(coords_with_intensity, max_t0, head)``; pass ``head`` back in for further
    timesteps of the same tree.
    """
    # 1. Get vessel tree (or reuse one whose dynamics were already computed, so that
    #    all timesteps of a tree share the same velocity jitter and arrival times)
    if head is None:
        head = coordinates2vessel(coords, interpolate=interp_coords_factor)
        head, max_t0 = compute_contrast_dynamics(
            head, add_gamma=True, V_base=V_base, P_flow=P_flow, alpha=alpha, beta=beta,
            branch_velocity_jitter=branch_velocity_jitter, dispersion_gain=dispersion_gain, rng=rng,
        )
        head.max_t0 = max_t0
    max_t0 = head.max_t0

    # 2. Intensities (I) for time t
    coords_new = coordinates_back(head, t)
    return coords_new, max_t0, head


def main():
    # Example usage for testing the new function
    # Note: The original main() loaded a file from disk;
    # but the full content of the file is not provided.
    
    # Placeholder for coordinates (e.g., a simple branching structure)
    coords_example = np.array([
        [0, 10, 20, 10, 20, 30, np.nan, 20, 30, np.nan],
        [0, 0, 0, 0, 10, 20, np.nan, 0, -10, np.nan],
        [0, 0, 0, 0, 0, 0, np.nan, 0, 0, np.nan],
        [5, 4, 3, 2, 2, 1, np.nan, 2, 1.5, np.nan] # Diameter d
    ])
    
    # New Gamma-variate parameters
    alpha = 4.0
    beta = 5.0
    V_base = 0.5  # Base flow velocity
    P_flow = 1.5  # Velocity exponent (d^1.5)
    
    t_start = 0
    t_end = 60
    
    # Example loop
    for t in range(t_start, t_end, 10):
        coords_with_intensity, max_t0, _ = bolus_injection(
            coords_example, t, alpha=alpha, beta=beta, V_base=V_base, P_flow=P_flow)
        print(f"Time t={t}, Max Arrival Time t0={max_t0:.2f}")
        # print("Output shape:", coords_with_intensity.shape)
        # print("First few points:\n", coords_with_intensity[:, :5])


if __name__ == "__main__":
    main()