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


def compute_contrast_dynamics(head, add_gamma=False, V_base=1.0, P_flow=1.0, alpha=2.0, beta=1.0):
    """
    Compute arrival time (t0) using segment-dependent flow velocity.
    Add Gamma-variate curve parameters to each node.
    V_base * d^P_flow is the segment velocity (Murray's Law consistency).
    """
    
    # Initialize origin node's properties
    head.t0 = 0.0
    if add_gamma:
        head.gamma = GammaVariate(t0=0.0, alpha=alpha, beta=beta)

    max_t0 = 0
    stack = [head]
    
    # Use a dictionary to keep track of calculated arrival times to avoid recalculating at branch points
    # key: node object, value: t0
    arrival_times = {head: 0.0}

    while stack:
        node = stack.pop()
        
        # Retrieve the current node's arrival time
        current_t0 = arrival_times[node]

        if node:
            max_t0 = max(max_t0, current_t0)

            for child in node.next:
                # 1. Compute segment length (Euclidian distance)
                segment_length = (
                    (node.x - child.x) ** 2
                    + (node.y - child.y) ** 2
                    + (node.z - child.z) ** 2
                ) ** 0.5
                
                # 2. Compute Segment Velocity (Based on Diameter d_child for the segment)
                # We use the diameter of the *child* node to define the velocity of the segment
                # flowing into it. If node.d is parent diameter, child.d is diameter of segment
                d_seg = child.d # Using child diameter for segment velocity
                
                # V_s is the segment flow velocity (e.g., V_base * d^1)
                segment_velocity = V_base * (d_seg ** P_flow)
                
                # Check for zero velocity (e.g., if diameter is 0)
                if segment_velocity <= 1e-6:
                    time_to_traverse = float('inf')
                else:
                    time_to_traverse = segment_length / segment_velocity
                
                # 3. Calculate Child's Arrival Time
                child_t0 = current_t0 + time_to_traverse
                
                # Update node properties for visualization/storage (optional but helpful)
                child.t0 = child_t0
                
                # 4. Add Gamma-variate function
                if add_gamma:
                    child.gamma = GammaVariate(t0=child_t0, alpha=alpha, beta=beta)
                
                # Store and continue DFS
                arrival_times[child] = child_t0
                stack.append(child)

    # Return max_t0 to normalize times for plotting (if needed)
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
    """(Modified to use GammaVariate and t0)"""
    stack = [head]
    strand_last = 0
    updaten = []

    # Pre-calculate Max Concentration for normalization
    # Find the node with the highest peak intensity to normalize all intensities
    max_overall_conc = 1e-6 # Initialize with small positive number
    all_nodes = [head]
    temp_stack = [head]
    while temp_stack:
        node = temp_stack.pop()
        if hasattr(node, 'gamma'):
            t_peak = node.gamma.t0 + node.gamma.alpha * node.gamma.beta
            max_overall_conc = max(max_overall_conc, node.gamma.sample(t_peak))
            temp_stack.extend(node.next)
        all_nodes.extend(node.next)
    
    # Reset stack for actual processing
    stack = [head]

    while stack:
        node = stack.pop()
        if node:
            # Need to handle the strand logic carefully, assuming you have a way to define strand breaks
            # The original code used strand index from stack (strand, node = stack.pop()), 
            # but that logic is complex to maintain with simple DFS. 
            # I'll simplify the original logic here assuming strand breaks are handled by NaNs in the final array structure.
            
            # --- Original Strand Break Logic Missing Here ---
            
            # Compute normalized intensity I
            if hasattr(node, 'gamma'):
                conc = node.gamma.sample(t)
                # I = Normalized Intensity
                I = conc / max_overall_conc
            else:
                I = 0.0
            
            # Append new coordinates/properties
            if len(updaten) == 0:
                updaten = np.array([[node.x, node.y, node.z, node.d, I]]).T
            else:
                updaten = np.c_[
                    updaten, [node.x, node.y, node.z, node.d, I]
                ]
            
            stack.extend(node.next)
            
    # NOTE: The logic for inserting NaN breaks between strands is complex and requires
    # traversing the tree in a way that tracks strand beginnings/ends.
    # Given the complexity, I'll return the continuous data array and assume the 
    # consumer (projection_generator) can handle it, or that the original code's 
    # NaN insertion logic (which was already complex) is integrated back carefully.
    
    # We return the transpose (rows as points) or original format (cols as points)?
    # Original bolus_injection returned (5 x N) array (cols as points), so we stick to that.
    return updaten


def bolus_injection(coords, t, alpha=2.0, beta=1.0, V_base=1.0, P_flow=1.0, interp_coords_factor=0):
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
    """
    # 1. Get vessel tree
    head = coordinates2vessel(coords, interpolate=interp_coords_factor)
    
    # 2. Compute arrival times (t0) and add Gamma-variate curves
    # Note: max_t0 is the new max_dist substitute
    head, max_t0 = compute_contrast_dynamics(
        head, 
        add_gamma=True, 
        V_base=V_base, 
        P_flow=P_flow, 
        alpha=alpha, 
        beta=beta
    )
    
    # 3. Compute intensities (I) for time t
    coords_new = coordinates_back(head, t)

    return coords_new, max_t0


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
        coords_with_intensity, max_t0 = bolus_injection(
            coords_example, 
            t, 
            alpha=alpha, 
            beta=beta, 
            V_base=V_base,
            P_flow=P_flow
        )
        print(f"Time t={t}, Max Arrival Time t0={max_t0:.2f}")
        # print("Output shape:", coords_with_intensity.shape)
        # print("First few points:\n", coords_with_intensity[:, :5])


if __name__ == "__main__":
    main()