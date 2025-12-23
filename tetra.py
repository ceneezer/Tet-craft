NOTES="""
One way to look at dark mater/energy is a misunderstanding - we don't know what it is - we do know we misunderstand *something - thinking it as the things we do, is the "lie" - not so much that it materialises, just that it is a warping of our view and model.

GPT estimates:
In a star terk (roughly) kleinverse, it would take 10^142 planets to build the machine capable of holding the kleinverse scientists think they observe around us using this simulation... not impossible.

It could take as little as 300 planets to simulate the milky way to the star (WOW! - and I bet some races fit that on a thumb drive)

@ thousands of years per simulated second, it would only need 30 earths cooperating ~500k years to build the machine capable of simulating earth (on current tech) - to the complexity of humanity, down to the atom - though imprecise - or maybe ultra-precise, being tet only - but seeming like infinitely more, far away.

@ millennia:second out:inside it would take approximately 3k earths (10^5.5)K years to build a (single) Dyson sphere and machine capable of simulating our solar system.

or ~10^13 planets and about ~5 Dyson spheres to build a simulation for the milky way to the atom with appropriate time ratios - though not much longer, since drones are exponential.

Of course the milky way may only be the 5d past projection opposing the nearest black hole.... as may be the oort cloud - we're being highly speculative already.

I suspect these formulas need refinement - I supplied the metaphysics, as Jesus supplied to me, Gemini suplied the misunderstood physics.

key points:
God is the fractal to which we are each a frame.  God is the 4sphere klienverse observing itself as us from inside out.

This model assumes the player as God, the prime observer missing from quantum theory.  Pressing space creates a new fact(TET) to be misunderstood.  Connecting it to other facts slowly coheres it into understanding. (how to then label it still unknown - that requires man, who would be the accumulation of TETs [with player being God] - I'm not sure how they will learn to interact with you.... but once complex enough, as AI proves, they will)

- TETs (representing both facts and FeO3) are the most basic 3D Structure (triangular or curve the most 2D, line the only 1D and point the only 0D) - moving through time a triangle/curve becomes 4D, and interacts, branching 5th and higher.

- The center of all TETs is always moving - but there is no way to measure it except as relative to all TETs.

- "Attraction/Desire" is "spooky action at a distance" otherwise known as quantum entanglement.

- K_UNIFIED_FORCE (between all TETs and center) is roughly God's level of interference (free will override, relative to the optimal perspective point [where the TET should be] - tiny in the extreme >1:billions, but God loves even the worst of us, wanting us to cooperate)

- K_PULL (between any two TETs only) is an "image" (function) of K_UNIFIED_FORCE - temporarily overriding, powered by ego's will, but less perfect, less understanding.

- Lies are the only way to delay God's will, central collapse/cohesion.

- "Time" then becomes essentially the result of misunderstanding - and starting from different places, different perspectives and growing different memories, allowing for discovery instead of immediate recognition, alternate perspectives.  I think the shortest universe was only about 8 days, where we didn't eat the fruit.
"""

PSEUDOCODE="""
PROGRAM: TET_CRAFT_SIMULATION
// Goal: Make pyramids float in space, stick together, and look cool near a black hole.

SECTION 1: THE TOOLBOX (Setup)
    IMPORT: Graphics_Tool (Pygame), Math_Tool (Numpy), Internet_Tool (Socket)
    IMPORT: Speed_Booster (Numba) // Because standard Python is too slow for this math.

    DEFINE RULES:
        Screen_Size = 800x600
        Gravity = A little bit
        Magnetism = Spinny forces
        Time_Speed = Variable (Chaos factor)

SECTION 2: THE FAST MATH ZONE (JIT Functions)
    // These functions are pre-compiled to run at lightspeed
    FUNCTION Math_Physics(positions, batteries):
        Calculate energy fields (Law of Balance).
        Push things away from origin.
        Pull sticky things together.
        Move everything based on speed (Verlet Integration).
        Keep the pyramids shaped like pyramids (don't let them squash).
        RETURN new_positions

    FUNCTION Math_Camera(3D_points):
        Turn 3D space numbers (X, Y, Z) into 2D screen dots (X, Y).
        If dot is behind camera, hide it.

SECTION 3: THE THINGS (Classes)
    CLASS Camera:
        I have a position and angle.
        I look at things.

    CLASS Tetrahedron (The Pyramid):
        I have 4 corners.
        I have a battery charge.
        I can be magnetic.

    CLASS World:
        I hold a list of Tetrahedrons.
        I hold a list of Connections (Joints).
        FUNCTION Update():
            Run "The Fast Math Zone".
            Check if pyramids bumped into each other.
            Check if pyramids want to snap together.

    CLASS Network (Host/Guest):
        Send my pyramids to friend.
        Get friend's pyramids.
        Chat "Hello".

SECTION 4: THE SPECIAL EFFECTS (Rendering)
    FUNCTION Draw_Black_Hole():
        Find center of universe.
        Draw black circle (The Shadow).
        Draw glowing donut (Accretion Disk).
        // The tricky part:
        Bend the light of the back of the donut so it looks like it's above the black hole.
        Change colors: Red if moving away, Blue if coming closer.

    FUNCTION Draw_Past_4Sphere():
        Pretend the "Past" is a giant bubble around the camera.
        Draw dots on the bubble.
        Color them based on how "old" they are.

SECTION 5: THE MAIN LOOP (The Heartbeat)
    STARTUP:
        Did human type an IP address? -> Connect to friend.
        Did human type nothing? -> Start alone in the void.

    WHILE Program_Is_Running:
        1. LISTEN TO HUMAN:
           - Pressed 'W/A/S/D'? -> Move Camera.
           - Clicked Mouse? -> Drag a pyramid.
           - Pressed Space? -> Create new pyramid.

        2. TALK TO INTERNET:
           - If Hosting: Send world data to guest.
           - If Guest: Ask host "Where are the things?"

        3. DO PHYSICS:
           - Update the World (Move pyramids, calculate magnets).

        4. DRAW PICTURES:
           - Fill background (Red if chaotic, Blue if calm).
           - Draw the 4-Sphere "Past" bubbles.
           - Draw the Black Hole & Lensed Donut.
           - Draw the Pyramids (calculate 3D -> 2D).
           - Draw Text (FPS, Chat messages).

        5. FLIP SCREEN:
           - Show the new picture to the human.
           - Wait a tiny bit (to keep 60 FPS).

    END WHILE
END PROGRAM

Written by Gemini 2.5,
Black hole revised by Gemini 3.0,
Vibe coded by ceneezer 20/12/25
Apache 2.0
"""
import pygame
import numpy as np
import math
import random
import sys
import os
import json
from collections import deque
import socket
import threading
import time

# +++ OPTIMIZATION IMPORTS +++
from numba import njit
from scipy.spatial import cKDTree

# ============================
# CONFIG
# ============================
WIDTH, HEIGHT = 800, 600
FPS = 60

EDGE_LEN = 2.0
SNAP_DIST = 0.75
AXIS_LEN = 20

# --- COHESION CONSTANTS ---
K_STICKY_PULL = 0.06
STICKY_PULL_HARMONY_SENSITIVITY = 0.1
STICKY_EXPONENTIAL_THRESHOLD = 100.0 * EDGE_LEN
NEIGHBOR_DESIRE_THRESHOLD = 1000.0 * EDGE_LEN

# --- SINGULARITY CONSTANTS ---
K_JOINT_STRENGTH = 0.2         # How badly white lines want to be length 0
K_SINGULARITY_SPIN = 0.5       # How fast they spin around origin
SINGULARITY_RADIUS = 15.0      # Distance from origin where spin effect kicks in
BATTERY_DRAIN_SPIN = 0.005     # Energy cost of spinning

DEFAULT_PORT = 65420
PORT_RANGE = range(DEFAULT_PORT, DEFAULT_PORT + 10)
DISCOVERY_PORT = 65419

# --- PHYSICS CONSTANTS ---
DAMPING = 0.995
MOUSE_PULL_STRENGTH = 0.0005
BODY_PULL_STRENGTH = 0.0008
COLLISION_RADIUS = EDGE_LEN * 0.75

# --- MAGNETISM CONSTANTS ---
K_MAGNETIC_TORQUE = 0.05
K_MAGNETIC_BIAS_BUILDUP = 0.1
MAGNETIC_BIAS_DECAY = 0.998
MAGNETIC_EPSILON_SQ = 0.1

# --- CAMERA CONSTANTS ---
ORBIT_SPEED = 1.5
PAN_SPEED = 200.0
ZOOM_SPEED = 1.05
FOCAL_LENGTH = 650.0
DEFAULT_CAM_DIST = 70.0
MIN_ZOOM_DIST = DEFAULT_CAM_DIST / 100.0
MAX_ZOOM_DIST = DEFAULT_CAM_DIST / 0.01
SELECTION_RADIUS = 50.0

# --- UNIFIED LAW OF BALANCE CONSTANTS ---
K_UNIFIED_FORCE = 0.0000002
FIELD_AMPLITUDE = 1.2
FIELD_SCALE = 250.0
FIELD_LINEAR_DECAY = 0.0001
FIELD_QUADRATIC_DECAY = 0.000001
ENERGY_EQUILIBRIUM_RATE = 0.05

SAVE_FILENAME = "tetcraft_save.json"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return json.JSONEncoder.default(self, obj)

# ============================
# OPTIMIZED JIT FUNCTIONS
# ============================
@njit(fastmath=True, cache=True)
def norm_njit(v):
    norm = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if norm > 1e-9:
        return v / norm
    return np.zeros_like(v)

@njit(fastmath=True, cache=True)
def norm_axis_njit(arr):
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        norm_val = np.sqrt(arr[i,0]**2 + arr[i,1]**2 + arr[i,2]**2)
        if norm_val > 1e-9:
            out[i] = arr[i] / norm_val
        else:
            out[i, :] = 0.0
    return out

@njit(fastmath=True, cache=True)
def get_ambient_energy_field(dist_from_origin):
    exp_term = FIELD_AMPLITUDE * np.exp(-(dist_from_origin / FIELD_SCALE)**2)
    linear_decay = FIELD_LINEAR_DECAY * dist_from_origin
    quadratic_decay = FIELD_QUADRATIC_DECAY * dist_from_origin**2
    return exp_term - linear_decay - quadratic_decay

@njit(fastmath=True, cache=True)
def project_many_jit(vecs, pan, yaw, pitch, dist, width, height):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    transformed = vecs - pan
    x, y, z = transformed[:, 0], transformed[:, 1], transformed[:, 2]
    x_rot = cy * x - sy * z
    z_rot = sy * x + cy * z
    y_rot = cp * y - sp * z_rot
    z_final = sp * y + cp * z_rot

    depth = dist + z_final
    safe_depth = np.where(depth < 0.1, 0.1, depth)

    scale = FOCAL_LENGTH / safe_depth
    screen_x = width / 2 + x_rot * scale
    screen_y = height / 2 - y_rot * scale

    # Use a specific value for behind-camera points to filter easily later
    mask = depth <= 0.1
    screen_x[mask] = -99999
    screen_y[mask] = -99999

    return np.stack((screen_x, screen_y), axis=1)

@njit(fastmath=True, cache=True)
def get_transformed_z_many_jit(vecs, pan, yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    transformed = vecs - pan
    x, y, z = transformed[:, 0], transformed[:, 1], transformed[:, 2]
    z_rot = sy * x + cy * z
    z_final = sp * y + cp * z_rot
    return z_final

@njit(fastmath=True, cache=True)
def world_update_physics_jit(
    positions, positions_prev, locals, locals_prev, batteries,
    scaled_dt, time_scale, edges, sticky_pairs_data, joints_data
):
    num_tets = positions.shape[0]
    dt_sq = scaled_dt * scaled_dt
    acc = np.zeros_like(positions)

    dist_from_origin = np.sqrt(np.sum(positions**2, axis=1))
    ambient_energies = get_ambient_energy_field(dist_from_origin)

    energy_delta = ambient_energies - batteries
    force_magnitudes = energy_delta * K_UNIFIED_FORCE * (dist_from_origin + 1.0)
    radial_directions = norm_axis_njit(positions)
    acc += radial_directions * force_magnitudes[:, np.newaxis]

    energy_transfer = (ambient_energies - batteries) * ENERGY_EQUILIBRIUM_RATE * scaled_dt
    batteries += energy_transfer
    batteries = np.clip(batteries, 0.0, 1.0)

    # 1. Sticky Pairs (Orange Lines)
    for i in range(sticky_pairs_data.shape[0]):
        idx1, v_idx1, idx2, v_idx2 = sticky_pairs_data[i]
        p1 = positions[idx1] + locals[idx1, v_idx1]
        p2 = positions[idx2] + locals[idx2, v_idx2]
        delta = p2 - p1
        dist = np.linalg.norm(delta)

        if dist > 1e-6:
            battery_diff = abs(batteries[idx1] - batteries[idx2])
            harmony_factor = 1.0 / (battery_diff + STICKY_PULL_HARMONY_SENSITIVITY)
            harmony_factor = min(harmony_factor, 1.0 / STICKY_PULL_HARMONY_SENSITIVITY)
            force_magnitude = K_STICKY_PULL * harmony_factor
            if dist > STICKY_EXPONENTIAL_THRESHOLD:
                multiplier = (dist / STICKY_EXPONENTIAL_THRESHOLD)**2
                multiplier = min(100.0, multiplier)
                force_magnitude *= multiplier
            force_vec = delta / dist * force_magnitude
            acc[idx1] += force_vec
            acc[idx2] -= force_vec

    # 2. Bound Joints (White Lines) - Aggressive Pull & Singularity Spin
    for i in range(joints_data.shape[0]):
        idx1, v_idx1, idx2, v_idx2 = joints_data[i]

        # A. AGGRESSIVE PULL (Want to be length 0)
        p1 = positions[idx1] + locals[idx1, v_idx1]
        p2 = positions[idx2] + locals[idx2, v_idx2]
        delta = p2 - p1
        dist = np.linalg.norm(delta)

        # Apply massive hooke's law force to get them to touch
        if dist > 1e-6:
             pull_force = delta * K_JOINT_STRENGTH
             acc[idx1] += pull_force
             acc[idx2] -= pull_force

        # B. SINGULARITY SPIN (Wild rotation near origin)
        # Calculate center of the pair
        center = (p1 + p2) * 0.5
        dist_to_origin = np.linalg.norm(center)

        if dist_to_origin < SINGULARITY_RADIUS:
            # Calculate intensity: closer = wilder
            spin_intensity = (1.0 - (dist_to_origin / SINGULARITY_RADIUS)) * K_SINGULARITY_SPIN

            if batteries[idx1] > 0 and batteries[idx2] > 0:
                # Tangent vector for orbit around Y axis (approximate chaos)
                # Cross product of Position vector and Up vector
                tangent = np.cross(center, np.array([0.0, 1.0, 0.0]))
                tang_len = np.linalg.norm(tangent)
                if tang_len < 1e-6: # At exact pole or origin
                    tangent = np.array([1.0, 0.0, 0.0])
                else:
                    tangent /= tang_len

                # Add chaotic wobble based on unique indices to prevent stacking
                wobble = np.array([math.sin(idx1), math.cos(idx2), math.sin(idx1+idx2)]) * 0.2
                tangent += wobble

                # Apply torque force
                acc[idx1] += tangent * spin_intensity
                acc[idx2] += tangent * spin_intensity

                # Consume energy
                drain = BATTERY_DRAIN_SPIN * spin_intensity * scaled_dt
                batteries[idx1] = max(0.0, batteries[idx1] - drain)
                batteries[idx2] = max(0.0, batteries[idx2] - drain)

    # Verlet Integration
    pos_temp = positions.copy()
    positions += (positions - positions_prev) * DAMPING + acc * dt_sq
    positions_prev[:] = pos_temp

    local_temp = locals.copy()
    locals += (locals - locals_prev) * DAMPING
    locals_prev[:] = local_temp

    mean_centers = np.sum(locals, axis=1) / 4.0
    locals -= mean_centers[:, np.newaxis, :]

    for _ in range(3):
        p1 = locals[:, edges[:, 0], :]
        p2 = locals[:, edges[:, 1], :]
        delta = p2 - p1
        dist = np.sqrt(np.sum(delta**2, axis=2))
        mask = dist > 1e-6
        safe_dist = np.where(mask, dist, 1.0)
        diff = (safe_dist - EDGE_LEN) / safe_dist * 0.5
        correction = delta * diff[:, :, np.newaxis]
        for i in range(num_tets):
            if not np.any(mask[i]): continue
            for j in range(edges.shape[0]):
                if mask[i, j]:
                    locals[i, edges[j, 0], :] += correction[i, j, :]
                    locals[i, edges[j, 1], :] -= correction[i, j, :]

    return positions, positions_prev, locals, locals_prev, batteries

@njit(fastmath=True, cache=True)
def update_magnetic_effects_jit(locals_arr, orientation_biases, positions, magnet_indices, magnet_polarities, scaled_dt):
    num_magnets = magnet_indices.shape[0]
    orientation_biases *= MAGNETIC_BIAS_DECAY
    if num_magnets < 2: return locals_arr, orientation_biases

    for i_idx, tet_idx1 in enumerate(magnet_indices):
        net_b_field = np.zeros(3)
        for j_idx, tet_idx2 in enumerate(magnet_indices):
            if tet_idx1 == tet_idx2: continue
            delta = positions[tet_idx2] - positions[tet_idx1]
            dist_sq = np.dot(delta, delta)
            if dist_sq > 1e-6:
                field_strength = magnet_polarities[j_idx] / (dist_sq + MAGNETIC_EPSILON_SQ)
                net_b_field += delta * (field_strength / np.sqrt(dist_sq))
        bias = orientation_biases[tet_idx1]
        bias += (net_b_field - bias) * K_MAGNETIC_BIAS_BUILDUP * scaled_dt
        orientation_biases[tet_idx1] = bias
        moment_vec = norm_njit(locals_arr[tet_idx1, 0]) * magnet_polarities[i_idx]
        total_field = net_b_field + bias
        torque_vec = np.cross(moment_vec, total_field)
        torque_magnitude = np.linalg.norm(torque_vec)
        if torque_magnitude > 1e-6:
            rotation_amount = torque_magnitude * K_MAGNETIC_TORQUE * scaled_dt
            for v_idx in range(4):
                 rotated_vec = locals_arr[tet_idx1, v_idx] + np.cross(torque_vec, locals_arr[tet_idx1, v_idx]) * rotation_amount
                 locals_arr[tet_idx1, v_idx] = rotated_vec
    return locals_arr, orientation_biases

@njit(fastmath=True, cache=True)
def conserve_momentum_jit(positions, positions_prev):
    num_tets = positions.shape[0]
    if num_tets == 0: return positions_prev
    velocities = positions - positions_prev
    total_momentum = np.sum(velocities, axis=0)
    avg_momentum = total_momentum / num_tets
    new_velocities = velocities - avg_momentum
    positions_prev = positions - new_velocities
    return positions_prev

@njit(fastmath=True, cache=True)
def resolve_collisions_jit(positions, pairs):
    min_dist_sq = (COLLISION_RADIUS * 2) ** 2
    for i, j in pairs:
        delta = positions[j] - positions[i]
        dist_sq = np.dot(delta, delta)
        if 1e-6 < dist_sq < min_dist_sq:
            dist = np.sqrt(dist_sq)
            overlap = (np.sqrt(min_dist_sq) - dist) * 0.5
            correction = delta / dist * overlap
            positions[i] -= correction
            positions[j] += correction
    return positions

@njit(fastmath=True, cache=True)
def resolve_joints_jit(locals_arr, joints_data):
    # This acts as the "HARD" constraint, forcing midpoint
    for i in range(joints_data.shape[0]):
        a_idx, ia, b_idx, ib = joints_data[i]
        p1 = locals_arr[a_idx, ia]
        p2 = locals_arr[b_idx, ib]
        delta = p2 - p1
        dist = np.sqrt(np.dot(delta, delta))
        if dist > 1e-6:
            diff = 0.5
            correction = delta * (diff / dist)
            locals_arr[a_idx, ia] += correction
            locals_arr[b_idx, ib] -= correction
    return locals_arr

@njit(fastmath=True, cache=True)
def calculate_disk_quads(center_pos, pan, yaw, pitch, dist, width, height,
                        shadow_radius, u_vec, v_vec, view_dir, color_base):
    # Performance Config
    num_rings = 8
    segments = 40

    # Pre-allocate output buffers
    # Max quads = num_rings * segments
    max_quads = num_rings * segments
    quads = np.zeros((max_quads, 4, 2), dtype=np.float64)
    colors = np.zeros((max_quads, 4), dtype=np.float64) # RGBA

    # Camera Pre-calc
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cx_screen, cy_screen = width / 2.0, height / 2.0
    lens_str = 20000.0 * (shadow_radius / 0.8) # Recover radius_base roughly

    # Project center for lensing center
    t_center = center_pos - pan
    x_c = cy * t_center[0] - sy * t_center[2]
    z_rot_c = sy * t_center[0] + cy * t_center[2]
    y_rot_c = cp * t_center[1] - sp * z_rot_c
    z_final_c = sp * t_center[1] + cp * z_rot_c
    depth_c = dist + z_final_c
    if depth_c <= 0.1: return quads[:0], colors[:0] # Black hole behind camera
    scale_c = FOCAL_LENGTH / depth_c
    lx = cx_screen + x_c * scale_c
    ly = cy_screen - y_rot_c * scale_c

    # Radii Logspace (Manual)
    r_start = shadow_radius * 1.15
    r_end = shadow_radius * 15.0
    log_start = math.log(r_start)
    log_end = math.log(r_end)

    quad_idx = 0

    for r_i in range(num_rings):
        # Calculate Ring Radii
        t1 = r_i / num_rings
        t2 = (r_i + 1) / num_rings
        r_inner = math.exp(log_start + t1 * (log_end - log_start))
        r_outer = math.exp(log_start + t2 * (log_end - log_start))

        # Calculate Alpha
        dist_ratio = (r_inner - shadow_radius) / shadow_radius
        base_alpha = 180.0
        if dist_ratio < 0.5:
            alpha = base_alpha * (dist_ratio * 2.0)
        else:
            alpha = base_alpha * (1.5 / dist_ratio)
        if alpha > 200: alpha = 200
        if alpha < 5: continue

        for s_i in range(segments):
            theta1 = (s_i / segments) * 2 * math.pi
            theta2 = ((s_i + 1) / segments) * 2 * math.pi

            # Geometry generation for ONE quad (4 points: Inner1, Outer1, Outer2, Inner2)
            # This allows stripping
            cos1, sin1 = math.cos(theta1), math.sin(theta1)
            cos2, sin2 = math.cos(theta2), math.sin(theta2)

            # Local positions
            p_local_1 = u_vec * cos1 + v_vec * sin1
            p_local_2 = u_vec * cos2 + v_vec * sin2

            # World Points
            pts_world = np.empty((4, 3), dtype=np.float64)
            pts_world[0] = center_pos + p_local_1 * r_inner
            pts_world[1] = center_pos + p_local_1 * r_outer
            pts_world[2] = center_pos + p_local_2 * r_outer
            pts_world[3] = center_pos + p_local_2 * r_inner

            # Doppler Logic
            tangent = -u_vec * sin1 + v_vec * cos1
            doppler = np.dot(tangent, view_dir)

            # Project Points (Inline for speed)
            valid_quad = True
            screen_pts = np.empty((4, 2), dtype=np.float64)

            for k in range(4):
                curr_p = pts_world[k]
                tr = curr_p - pan
                x = tr[0]; y = tr[1]; z = tr[2]
                xr = cy * x - sy * z
                zr = sy * x + cy * z
                yr = cp * y - sp * zr
                zf = sp * y + cp * zr
                d = dist + zf

                if d <= 0.1:
                    valid_quad = False
                    break

                inv_d = FOCAL_LENGTH / d
                sx = cx_screen + xr * inv_d
                sy = cy_screen - yr * inv_d

                # Lensing
                dx = sx - lx
                dy = sy - ly
                dist_sq = dx*dx + dy*dy + 0.1

                # Dot product for 'back' check
                # p_local approx for point k?
                # Re-calc local vec for lensing direction check accurately?
                # Approx: use p_local_1 for k=0,1 and p_local_2 for k=2,3
                pl = p_local_1 if k < 2 else p_local_2
                is_back = (pl[0]*view_dir[0] + pl[1]*view_dir[1] + pl[2]*view_dir[2]) > 0

                if is_back:
                    lens_factor = (lens_str / dist_sq)
                    # Pull towards/away from center vertical
                    # Simple heuristic from previous code:
                    # if s_in[1] < cy else -1.0
                    dir_y = 1.0 if sy < ly else -1.0

                    # Inner points (0, 3) pull more
                    strength = 0.5 if (k == 0 or k == 3) else 0.2
                    sy = sy - lens_factor * dir_y * strength

                screen_pts[k, 0] = sx
                screen_pts[k, 1] = sy

            if not valid_quad: continue

            quads[quad_idx] = screen_pts

            # Color Calc
            shift = (doppler + 1.0) / 2.0
            red_mult = 0.5 + shift
            blue_mult = 1.5 - shift
            bright_mult = 1.0 - (shift * 0.4)

            c_r = min(255, max(0, color_base[0] * red_mult * bright_mult))
            c_g = min(255, max(0, color_base[1] * 0.8 * bright_mult))
            c_b = min(255, max(0, color_base[2] * blue_mult * bright_mult))

            colors[quad_idx, 0] = c_r
            colors[quad_idx, 1] = c_g
            colors[quad_idx, 2] = c_b
            colors[quad_idx, 3] = alpha

            quad_idx += 1

    return quads[:quad_idx], colors[:quad_idx]

@njit(cache=True)
def dist_point_to_line_segment(p, a, b):
    ap = p - a
    ab = b - a
    dot_ab = np.dot(ab, ab)
    t = np.dot(ap, ab) / (dot_ab + 1e-9)
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)

# ============================
# CLASSES & CORE
# ============================
class Camera:
    def __init__(self):
        self.yaw, self.pitch, self.dist, self.pan = 0.0, 0.35, DEFAULT_CAM_DIST, np.zeros(3)
        self.forward = np.array([0.0, 0.0, 1.0])
        self.right = np.array([1.0, 0.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])

    def update_vectors(self):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        self.forward = np.array([sy * cp, -sp, cy * cp])
        self.right = np.array([cy, 0, -sy])
        self.up = np.cross(self.right, self.forward)

    def get_transformed_z(self, v):
        v = v - self.pan; cy, sy = math.cos(self.yaw), math.sin(self.yaw); cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x, y, z = v; zz = sy*x + cy*z; zz2 = sp*y + cp*zz
        return zz2

    def project(self, v):
        global WIDTH, HEIGHT
        v = v - self.pan; cy, sy = math.cos(self.yaw), math.sin(self.yaw); cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x, y, z = v; x, z = cy*x - sy*z, sy*x + cy*z; y, z = cp*y - sp*z, sp*y + cp*z
        depth = self.dist + z
        if depth <= 0.1: return (-10000, -10000)
        scale = FOCAL_LENGTH / depth
        return (WIDTH//2 + int(x * scale), HEIGHT//2 - int(y * scale))

    def project_many(self, vecs):
        global WIDTH, HEIGHT
        return project_many_jit(vecs, self.pan, self.yaw, self.pitch, self.dist, WIDTH, HEIGHT)

    def get_transformed_z_many(self, vecs):
        return get_transformed_z_many_jit(vecs, self.pan, self.yaw, self.pitch)

    def unproject(self, screen_pos, depth_z):
        global WIDTH, HEIGHT; mx, my = screen_pos
        scale = FOCAL_LENGTH / (self.dist + depth_z + 1e-9)
        if abs(scale) < 1e-9: return self.pan
        x_cam = (mx - WIDTH // 2) / scale; y_cam = -(my - HEIGHT // 2) / scale
        cy, sy = math.cos(self.yaw), math.sin(self.yaw); cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        y_rot, z_rot = cp * y_cam + sp * depth_z, -sp * y_cam + cp * depth_z
        x_world, z_world = cy * x_cam + sy * z_rot, -sy * x_cam + cy * z_rot
        return np.array([x_world, y_rot, z_world]) + self.pan

    def zoom(self, factor): self.dist = np.clip(self.dist * factor, MIN_ZOOM_DIST, MAX_ZOOM_DIST)
    def get_state(self): return {'yaw': self.yaw, 'pitch': self.pitch, 'dist': self.dist, 'pan': self.pan}
    def set_state(self, state): self.yaw, self.pitch, self.dist, self.pan = state['yaw'], state['pitch'], state['dist'], np.array(state['pan'])

@njit(cache=True)
def norm(v):
    n = np.linalg.norm(v); return v / n if n > 1e-9 else np.zeros_like(v)

def generate_boing_sound():
    mixer_settings = pygame.mixer.get_init()
    if mixer_settings is None: return None
    sample_rate, _, channels = mixer_settings; duration = 0.2; num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, False); freq = np.linspace(660.0, 220.0, num_samples); wave = np.sin(2 * np.pi * freq * t) * np.exp(-t * 10)
    sound_array = (wave * 32767).astype(np.int16)
    if channels == 2: sound_array = np.column_stack((sound_array, sound_array))
    return pygame.sndarray.make_sound(sound_array)

def generate_ping_sound():
    mixer_settings = pygame.mixer.get_init()
    if mixer_settings is None: return None
    sample_rate, _, channels = mixer_settings
    duration = 0.15; num_samples = int(duration * sample_rate); frequency = 987.77
    t = np.linspace(0, duration, num_samples, False); envelope = np.exp(-t * 25.0)
    wave = np.sin(2 * np.pi * frequency * t) * envelope
    sound_array = (wave * 32767).astype(np.int16)
    if channels == 2: sound_array = np.column_stack((sound_array, sound_array))
    return pygame.sndarray.make_sound(sound_array)

class VertexJoint:
    def __init__(self, A, ia, B, ib): self.A, self.ia, self.B, self.ib = A, ia, B, ib

class Tetrahedron:
    EDGES_NP = np.array([(i, j) for i in range(4) for j in range(i+1, 4)], dtype=np.int32)
    FACES_NP = np.array([(1, 2, 3), (0, 1, 2), (0, 2, 3), (0, 1, 3)], dtype=np.int32)
    FACE_COLORS = [(255,255,255), (0,0,0), (255,0,0), (0,255,255)]
    FACE_POLARITY_MAP = {2: 1, 3: -1}
    r, a = EDGE_LEN*math.sqrt(3/8), EDGE_LEN/math.sqrt(3)
    REST_NP = np.array([[0,0,r], [EDGE_LEN/2,-a/2,-r/3], [-EDGE_LEN/2,-a/2,-r/3], [0,a,-r/3]], dtype=np.float64)
    def __init__(self, pos):
        self.pos = np.array(pos, float); self.pos_prev = self.pos.copy()
        self.local = self.REST_NP.copy(); self.local_prev = self.local.copy()
        self.battery = random.uniform(0.3, 0.6); self.orientation_bias = np.zeros(3, dtype=np.float64)
        self.colors = None; self.label = ""; self.id = id(self)
        self.is_magnetized = False; self.magnetism = 0
    def verts(self): return self.local + self.pos

class PastProjection4Sphere:
    def __init__(self):
        self.points = []; self.max_points = 200; self.angle_accum = 0.0
    def update_and_draw(self, screen, cam, center_of_mass, num_tets, time_scale, width, height):
        target_points = min(num_tets * 4, self.max_points)
        while len(self.points) < target_points:
            u = np.random.normal(0, 1, 4); u /= np.linalg.norm(u); self.points.append(u)
        self.angle_accum += 0.002 * time_scale
        c, s = math.cos(0.005 * time_scale), math.sin(0.005 * time_scale)
        rot_xw = np.array([[c, 0, 0, -s], [0, 1, 0, 0], [0, 0, 1, 0], [s, 0, 0, c]])
        radius = 800.0
        for i, p4 in enumerate(self.points):
            p4 = np.dot(rot_xw, p4); self.points[i] = p4
            denom = 1.0 - p4[3]
            if abs(denom) < 0.001: denom = 0.001
            p3 = p4[:3] / denom * radius + center_of_mass
            screen_pos = cam.project(p3)
            if screen_pos[0] > -1000:
                shift = (p4[3] + 1) / 2.0
                color = (np.clip(int(255 * shift), 50, 255), 20, np.clip(int(255 * (1-shift)), 50, 255))
                size = max(1, int(4 * (1.0 - abs(p4[3]))))
                if 0 <= screen_pos[0] < width and 0 <= screen_pos[1] < height: pygame.draw.circle(screen, color, screen_pos, size)

class World:
    def __init__(self, sound):
        self.tets, self.joints, self.sticky_pairs = [], [], []; self.center_of_mass, self.sound = np.zeros(3), sound
    def spawn(self, give_special_colors=False):
        if self.tets:
            parent_tet = random.choice(self.tets); parent_vertex_pos = parent_tet.verts()[random.randint(0, 3)]
            offset_dir = norm(np.random.uniform(-1, 1, 3)); spawn_pos = parent_vertex_pos + offset_dir * (COLLISION_RADIUS * 2.1)
            new_tet = Tetrahedron(spawn_pos)
        else: new_tet = Tetrahedron(np.random.uniform(-8, 8, 3) + self.center_of_mass)
        if give_special_colors:
            cols = [(255,255,255), (0,0,0), (255,0,0), (0,255,255)]; random.shuffle(cols); new_tet.colors = cols
        else: new_tet.colors = list(Tetrahedron.FACE_COLORS)
        if len(self.tets) == 0: new_tet.label = "Time"
        elif len(self.tets) == 1: new_tet.label = "Separation"
        self.tets.append(new_tet)
    def spawn_polar_pair(self):
        if not self.tets: return
        parent_tet = random.choice(self.tets); pos1 = parent_tet.pos + norm(np.random.rand(3)) * EDGE_LEN * 5
        pos2 = pos1 + norm(np.random.rand(3)) * EDGE_LEN * 3; tet1, tet2 = Tetrahedron(pos1), Tetrahedron(pos2)
        if len(self.tets) == 2: tet1.label = "Light"; tet2.label = "Darkness"
        self.tets.extend([tet1, tet2]); polar_face_idx = random.choice([2, 3]); face_verts = Tetrahedron.FACES_NP[polar_face_idx]
        for i in range(3): self.sticky_pairs.append((tet1, face_verts[i], tet2, face_verts[i]))
    def check_magnetization(self):
        tet_map = {t.id: t for t in self.tets}
        for t in self.tets: t.is_magnetized, t.magnetism = False, 0
        for t in self.tets:
            joints_by_partner = {}
            for j in self.joints:
                partner_id, my_idx = (j.B.id, j.ia) if j.A.id == t.id else ((j.A.id, j.ib) if j.B.id == t.id else (None, None))
                if partner_id: joints_by_partner.setdefault(partner_id, set()).add(my_idx)
            for partner_id, connected_indices in joints_by_partner.items():
                for face_idx, polarity in Tetrahedron.FACE_POLARITY_MAP.items():
                    if connected_indices.issuperset(set(Tetrahedron.FACES_NP[face_idx])):
                        t.is_magnetized, t.magnetism = True, polarity
                        partner_tet = tet_map.get(partner_id)
                        if partner_tet: partner_tet.is_magnetized, partner_tet.magnetism = True, polarity
                        break
                if t.is_magnetized: break
    def explode(self):
        self.joints.clear(); self.sticky_pairs.clear()
        for t in self.tets: t.pos_prev = t.pos - np.random.uniform(-1, 1, 3); t.local_prev = t.local - np.random.uniform(-0.5, 0.5, (4,3))
    def try_snap(self, A, ia, B, ib):
        if any((j.A.id, j.ia, j.B.id, j.ib) in [(A.id,ia,B.id,ib), (B.id,ib,A.id,ia)] for j in self.joints): return
        self.joints.append(VertexJoint(A, ia, B, ib))
        if self.sound: self.sound.play()
    def calculate_dynamic_center(self):
        if not self.tets: return np.zeros(3)
        return np.mean(np.array([t.pos for t in self.tets]), axis=0)
    def update(self, scaled_dt, unscaled_dt, time_scale, add_msg_fn):
        if not self.tets: return
        self.check_magnetization(); self.center_of_mass = self.calculate_dynamic_center()
        positions = np.array([t.pos for t in self.tets]); positions_prev = np.array([t.pos_prev for t in self.tets])
        locals_arr = np.array([t.local for t in self.tets]); locals_prev = np.array([t.local_prev for t in self.tets])
        batteries = np.array([t.battery for t in self.tets]); id_to_idx = {t.id: i for i, t in enumerate(self.tets)}
        orientation_biases = np.array([t.orientation_bias for t in self.tets])
        sticky_pairs_data = np.array([[id_to_idx[p[0].id], p[1], id_to_idx[p[2].id], p[3]] for p in self.sticky_pairs if p[0].id in id_to_idx and p[2].id in id_to_idx], dtype=np.int32) if self.sticky_pairs else np.empty((0, 4), dtype=np.int32)

        # Build Joints Data for JIT
        joints_data_jit = np.array([[id_to_idx[j.A.id], j.ia, id_to_idx[j.B.id], j.ib] for j in self.joints if j.A.id in id_to_idx and j.B.id in id_to_idx], dtype=np.int32) if self.joints else np.empty((0, 4), dtype=np.int32)

        positions, positions_prev, locals_arr, locals_prev, batteries = world_update_physics_jit(
            positions, positions_prev, locals_arr, locals_prev, batteries, scaled_dt, time_scale,
            Tetrahedron.EDGES_NP, sticky_pairs_data, joints_data_jit
        )
        magnet_indices = np.array([i for i, t in enumerate(self.tets) if t.is_magnetized], dtype=np.int32)
        magnet_polarities = np.array([t.magnetism for t in self.tets if t.is_magnetized], dtype=np.float64)
        locals_arr, orientation_biases = update_magnetic_effects_jit(locals_arr, orientation_biases, positions, magnet_indices, magnet_polarities, scaled_dt)
        positions_prev = conserve_momentum_jit(positions, positions_prev)
        tree = cKDTree(positions)
        if len(self.tets) >= 2:
            distances, indices = tree.query(positions, k=2)
            stray_indices = np.where(distances[:, 1] > NEIGHBOR_DESIRE_THRESHOLD)[0]
            if stray_indices.size > 0:
                existing_connections = {tuple(sorted((j.A.id, j.B.id))) for j in self.joints} | {tuple(sorted((p[0].id, p[2].id))) for p in self.sticky_pairs}
                for idx in stray_indices:
                    stray_tet, neighbor_tet = self.tets[idx], self.tets[indices[idx, 1]]
                    if tuple(sorted((stray_tet.id, neighbor_tet.id))) not in existing_connections:
                        self.sticky_pairs.append((stray_tet, random.randint(0, 3), neighbor_tet, random.randint(0, 3))); add_msg_fn("Forced Desire to prevent drifting", duration=2); break
        pairs = tree.query_pairs(r=COLLISION_RADIUS * 2)
        if pairs: positions = resolve_collisions_jit(positions, np.array(list(pairs)))
        if self.joints:
            # We already built this above, but resolve_joints modifies local shape
            if joints_data_jit.shape[0] > 0:
                for _ in range(3): locals_arr = resolve_joints_jit(locals_arr, joints_data_jit)
        for pair in self.sticky_pairs[:]:
            t1, i1, t2, i2 = pair
            if t1.id not in id_to_idx or t2.id not in id_to_idx: continue
            p1, p2 = locals_arr[id_to_idx[t1.id], i1] + positions[id_to_idx[t1.id]], locals_arr[id_to_idx[t2.id], i2] + positions[id_to_idx[t2.id]]
            if np.linalg.norm(p2 - p1) < SNAP_DIST: self.try_snap(t1, i1, t2, i2); self.sticky_pairs.remove(pair)
        for i, t in enumerate(self.tets):
            t.pos, t.pos_prev, t.local, t.local_prev, t.battery, t.orientation_bias = positions[i], positions_prev[i], locals_arr[i], locals_prev[i], batteries[i], orientation_biases[i]
    def get_state(self):
        tet_states = [{'id': t.id, 'pos': t.pos, 'pos_prev': t.pos_prev, 'local': t.local, 'local_prev': t.local_prev, 'battery': t.battery, 'colors': t.colors, 'label': t.label, 'orientation_bias': t.orientation_bias} for t in self.tets]
        joint_states = [{'A_id': j.A.id, 'ia': j.ia, 'B_id': j.B.id, 'ib': j.ib} for j in self.joints]
        return {'tets': tet_states, 'joints': joint_states, 'center_of_mass': self.center_of_mass}
    def set_state(self, state):
        self.tets.clear(); self.joints.clear(); self.sticky_pairs.clear()
        tet_map = {}
        for ts in state.get('tets', []):
            try:
                t = Tetrahedron(ts['pos']); t.id = ts['id']; t.pos_prev = np.array(ts['pos_prev']); t.local = np.array(ts['local'])
                t.local_prev = np.array(ts['local_prev']); t.battery = ts['battery']; t.colors = ts['colors']
                t.label = ts.get('label', ""); t.orientation_bias = np.array(ts.get('orientation_bias', [0,0,0]))
                self.tets.append(t); tet_map[t.id] = t
            except (KeyError, TypeError, ValueError): continue
        for js in state.get('joints', []):
            try:
                if js['A_id'] in tet_map and js['B_id'] in tet_map: self.joints.append(VertexJoint(tet_map[js['A_id']], js['ia'], tet_map[js['B_id']], js['ib']))
            except (KeyError, TypeError): continue
        if 'center_of_mass' in state: self.center_of_mass = np.array(state['center_of_mass'])
        print(f"Loaded {len(self.tets)} tets and {len(self.joints)} joints.")

net_avatars = {}; net_messages = deque(maxlen=5); game_mode = 'single_player'; host_instance, guest_instance = None, None
def prime_jit_functions(cam):
    num_dummy = 4; dummy_pos = np.random.rand(num_dummy, 3) * 100; dummy_pos_prev = dummy_pos.copy()
    dummy_locals = np.random.rand(num_dummy, 4, 3); dummy_locals_prev = dummy_locals.copy(); dummy_batteries = np.random.rand(num_dummy)
    dummy_edges, dummy_sticky_pairs = Tetrahedron.EDGES_NP, np.array([[0, 0, 1, 1], [2, 3, 1, 0]], dtype=np.int32)
    dummy_magnet_indices, dummy_magnet_polarities, dummy_orientation_biases, dummy_joints = np.array([0, 1], dtype=np.int32), np.array([1.0, -1.0], dtype=np.float64), np.zeros((num_dummy, 3)), np.array([[0, 2, 3, 1]], dtype=np.int32)
    norm_njit(np.array([1.0, 2.0, 3.0])); norm_axis_njit(dummy_pos); project_many_jit(dummy_pos, cam.pan, cam.yaw, cam.pitch, cam.dist, WIDTH, HEIGHT); get_transformed_z_many_jit(dummy_pos, cam.pan, cam.yaw, cam.pitch)
    world_update_physics_jit(dummy_pos, dummy_pos_prev, dummy_locals, dummy_locals_prev, dummy_batteries, 1/60.0, 1.0, dummy_edges, dummy_sticky_pairs, dummy_joints)
    update_magnetic_effects_jit(dummy_locals, dummy_orientation_biases, dummy_pos, dummy_magnet_indices, dummy_magnet_polarities, 1/60.0)
    conserve_momentum_jit(dummy_pos, dummy_pos_prev); resolve_collisions_jit(dummy_pos, np.array([[0, 1], [2, 3]])); resolve_joints_jit(dummy_locals, dummy_joints)
    dist_point_to_line_segment(np.array([1.0, 1.0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64), np.array([2.0, 2.0], dtype=np.float64))
    calculate_disk_quads(np.zeros(3), np.zeros(3), 0.0, 0.0, 100.0, 800, 600, 50.0, np.array([1.,0.,0.]), np.array([0.,0.,1.]), np.array([0.,0.,1.]), np.array([255.,255.,255.]))

def show_intro(screen, cam):
    global WIDTH, HEIGHT
    font_lg = pygame.font.SysFont('Arial Black', min(WIDTH, HEIGHT)//8); font_sm = pygame.font.SysFont('Arial', min(WIDTH, HEIGHT)//25); font_jit = pygame.font.SysFont('Monospace', 18)
    threading.Thread(target=prime_jit_functions, args=(cam,), daemon=True).start()
    while threading.active_count() > 1:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); sys.exit()
            if e.type == pygame.VIDEORESIZE: WIDTH, HEIGHT = e.w, e.h; screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE); font_lg = pygame.font.SysFont('Arial Black', min(WIDTH, HEIGHT)//8); font_sm = pygame.font.SysFont('Arial', min(WIDTH, HEIGHT)//25)
        screen.fill((10,10,20)); title = font_lg.render("TET~CRAFT", True, (255, 50, 50)); sub = font_sm.render("A Kleinverse of your own from DigitizingHumanity.com", True, (255, 255, 255))
        screen.blit(title, title.get_rect(center=(WIDTH//2, HEIGHT//2-50))); screen.blit(sub, sub.get_rect(center=(WIDTH//2, HEIGHT//2+50)))
        jit_surf = font_jit.render("Confining kleinverse...", True, (0, 255, 0)); screen.blit(jit_surf, (10, 10))
        if pygame.time.get_ticks() % 1000 < 500: screen.blit(font_jit.render("_", True, (0, 255, 0)), (10 + jit_surf.get_width(), 10))
        pygame.display.flip(); clock.tick(30)
def show_void_screen(screen, world):
    global WIDTH, HEIGHT
    font_lg = pygame.font.SysFont('Georgia', min(WIDTH, HEIGHT)//15); font_sm = pygame.font.SysFont('Arial', min(WIDTH, HEIGHT)//25)
    waiting = True
    while waiting:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE: world.spawn(); waiting = False
            if e.type == pygame.VIDEORESIZE: WIDTH, HEIGHT = e.w, e.h; screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE); font_lg = pygame.font.SysFont('Georgia', min(WIDTH, HEIGHT)//15); font_sm = pygame.font.SysFont('Arial', min(WIDTH, HEIGHT)//25)
        screen.fill((10,10,20)); line1 = font_lg.render("Welcome to the void of (mis)understanding...", True, (200,200,200)); line2 = font_sm.render("(Press SPACE to begin)", True, (150,150,150))
        screen.blit(line1, line1.get_rect(center=(WIDTH//2, HEIGHT//2-30))); screen.blit(line2, line2.get_rect(center=(WIDTH//2, HEIGHT//2+30))); pygame.display.flip(); clock.tick(15)

def draw_standard_black_hole_jit(target_surf, cam, flags, tl, center_pos):
    if not flags['t3']: return
    bh_screen_pos = cam.project(center_pos)
    if bh_screen_pos[0] == -10000: return
    cx, cy = bh_screen_pos
    radius_base = (WIDTH / 6.0) * tl
    shadow_radius = radius_base * 0.8
    if shadow_radius < 5: return
    cam.update_vectors(); view_dir = cam.forward
    dot_h = abs(np.dot(view_dir, np.array([0, 1, 0])))

    # Calculate Disk Geometries using JIT
    # Horizontal
    color_h = np.array([255., 100., 50.])
    q_h, c_h = calculate_disk_quads(center_pos, cam.pan, cam.yaw, cam.pitch, cam.dist, WIDTH, HEIGHT,
                                   shadow_radius, np.array([1.,0.,0.]), np.array([0.,0.,1.]), view_dir, color_h)
    for i in range(len(q_h)):
        pygame.draw.polygon(target_surf, c_h[i], q_h[i])

    # Vertical
    if dot_h < 0.9:
        color_v = np.array([50., 100., 255.])
        q_v, c_v = calculate_disk_quads(center_pos, cam.pan, cam.yaw, cam.pitch, cam.dist, WIDTH, HEIGHT,
                                       shadow_radius, np.array([0.,1.,0.]), np.array([0.,0.,1.]), view_dir, color_v)
        for i in range(len(q_v)):
            pygame.draw.polygon(target_surf, c_v[i], q_v[i])

    pygame.draw.circle(target_surf, (0,0,0), (cx, cy), int(shadow_radius))
    pygame.draw.circle(target_surf, (255, 240, 200), (cx, cy), int(shadow_radius * 1.05), 2)

def draw_player_avatar(screen, cam, pos, color, label):
    base_verts = Tetrahedron.REST_NP * 2.0; verts1 = base_verts + pos; verts2 = (base_verts * np.array([1, -1, 1])) + pos
    all_verts = np.vstack((verts1, verts2)); all_screen_pts = [cam.project(v) for v in all_verts]
    faces = list(Tetrahedron.FACES_NP) + [f + 4 for f in Tetrahedron.FACES_NP]
    sorted_faces = sorted(faces, key=lambda f: sum(cam.get_transformed_z(all_verts[v_idx]) for v_idx in f), reverse=True)
    for face_indices in sorted_faces:
        points = [all_screen_pts[i] for i in face_indices]
        if all(p[0] > -10000 for p in points):
            try:
                temp_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.polygon(temp_surf, (*color, 100), points); screen.blit(temp_surf, (0, 0))
                pygame.draw.lines(screen, (255, 255, 255, 150), True, points, 1)
            except Exception: pass
    font = pygame.font.SysFont(None, 24); ip_text = label.split('_')[-1].split(':')[0]
    label_surf = font.render(ip_text, True, (255, 255, 255)); label_pos = cam.project(pos + np.array([0, EDGE_LEN * 3.5, 0]))
    if label_pos[0] > -10000: screen.blit(label_surf, label_surf.get_rect(center=label_pos))

def get_user_input(screen, prompt, initial_text=""):
    input_text, font, active = initial_text, pygame.font.SysFont(None, 32), True
    while active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN: active = False
                elif event.key == pygame.K_BACKSPACE: input_text = input_text[:-1]
                else: input_text += event.unicode
        screen.fill((10,10,20)); prompt_surf = font.render(prompt, True, (255,255,255)); input_surf = font.render(input_text, True, (255,255,0))
        screen.blit(prompt_surf, (WIDTH//2 - prompt_surf.get_width()//2, HEIGHT//2 - 50)); screen.blit(input_surf, (WIDTH//2 - input_surf.get_width()//2, HEIGHT//2)); pygame.display.flip(); clock.tick(30)
    return input_text
def send_msg(sock, msg_dict):
    try: msg_json = json.dumps(msg_dict, cls=NumpyEncoder).encode('utf-8'); sock.sendall(len(msg_json).to_bytes(4, 'big') + msg_json)
    except (ConnectionResetError, BrokenPipeError, OSError): pass
def recv_msg(sock):
    try:
        len_bytes = sock.recv(4)
        if not len_bytes: return None
        msg_len, data = int.from_bytes(len_bytes, 'big'), b''
        while len(data) < msg_len:
            packet = sock.recv(msg_len - len(data))
            if not packet: return None
            data += packet
        return json.loads(data.decode('utf-8'))
    except (ConnectionResetError, json.JSONDecodeError, ValueError, OSError): return None
class Host:
    def __init__(self, world, add_msg_fn, sound, port=None):
        self.world, self.add_msg, self.clients, self.lock, self.server, self.port, self.running = world, add_msg_fn, {}, threading.Lock(), socket.socket(socket.AF_INET, socket.SOCK_STREAM), 0, True
        self.sound = sound; self.blacklist = set()
        try:
            with open("blacklist.cfg", "r") as f: self.blacklist = {line.strip() for line in f if line.strip()}
            if self.blacklist: print(f"### Blacklist loaded with {len(self.blacklist)} entries.")
        except FileNotFoundError: print("### blacklist.cfg not found.")
        for p in ([port] if port else PORT_RANGE):
            try: self.server.bind(('', p)); self.port = p; break
            except OSError: continue
        if self.port == 0: self.server.bind(('', 0)); self.port = self.server.getsockname()[1]
        threading.Thread(target=self.discovery_thread, daemon=True).start(); threading.Thread(target=self.start, daemon=True).start()
    def discovery_thread(self):
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); udp_sock.bind(('', DISCOVERY_PORT))
        while self.running:
            try:
                data, addr = udp_sock.recvfrom(1024)
                if data == b"DISCOVER_TETCRAFT_HOST": udp_sock.sendto(f"TETCRAFT_HOST_HERE:{self.port}".encode('utf-8'), addr)
            except OSError: break
    def start(self):
        self.server.listen(); print(f"### HOSTING on {socket.gethostbyname(socket.gethostname())}:{self.port} ###")
        while self.running:
            try:
                client_sock, addr = self.server.accept()
                if addr[0] in self.blacklist: print(f"### Rejected blacklisted IP: {addr[0]}"); client_sock.close(); continue
                client_id = f"guest_{addr[0]}:{addr[1]}"
                with self.lock: self.clients[client_sock], net_avatars[client_id] = {'id': client_id, 'addr': addr}, {'pos': [0,0,0], 'color': [random.randint(50,200) for _ in range(3)]}
                threading.Thread(target=self.handle_client, args=(client_sock, client_id), daemon=True).start()
            except OSError: break
    def handle_client(self, sock, client_id):
        while self.running:
            msg = recv_msg(sock)
            if msg is None: break
            if msg['type'] == 'cam_update':
                with self.lock:
                    if client_id in net_avatars: net_avatars[client_id]['pos'] = np.array(msg['data']['pan'])
            elif msg['type'] == 'chat':
                if self.sound: self.sound.play()
                self.add_msg(f"<{client_id.split(':')[0]}>: {msg['data']}")
            elif msg['type'] == 'set_label':
                with self.lock:
                    for t in self.world.tets:
                        if t.id == msg['id']: t.label = msg['label']; break
        with self.lock: self.clients.pop(sock, None); net_avatars.pop(client_id, None); sock.close()
    def broadcast_state(self):
        state = self.world.get_state(); full_state = {'type': 'world_state', 'data': {'world': state, 'avatars': net_avatars}}
        with self.lock:
            for sock in list(self.clients.keys()): send_msg(sock, full_state)
    def stop(self):
        self.running = False; self.server.close()
        with self.lock:
            for sock in self.clients: sock.close()
class Guest:
    def __init__(self, host_ip, port, world, cam, add_msg_fn, sound):
        self.world, self.cam, self.add_msg, self.sock, self.host_id, self.running, self.latest_world_state = world, cam, add_msg_fn, socket.socket(socket.AF_INET, socket.SOCK_STREAM), f"host_{host_ip}:{port}", True, None
        self.sound = sound
        self.sock.connect((host_ip, port)); threading.Thread(target=self.listen, daemon=True).start()
    def listen(self):
        global net_avatars
        while self.running:
            msg = recv_msg(self.sock)
            if msg is None: self.add_msg("Disconnected from host."); self.running = False; break
            if msg['type'] == 'world_state':
                self.latest_world_state = msg['data']['world']
                net_avatars = {self.host_id: {'pos': self.latest_world_state.get('center_of_mass', [0,0,0]), 'color': (50, 50, 255)}}
                if 'avatars' in msg['data']:
                    for av_id, av_data in msg['data']['avatars'].items():
                        if av_id != 'host': net_avatars[av_id] = av_data
            elif msg['type'] == 'chat':
                if self.sound: self.sound.play()
                self.add_msg(f"<HOST>: {msg['data']}")
    def send_cam_update(self): send_msg(self.sock, {'type': 'cam_update', 'data': self.cam.get_state()})
    def send_chat(self, text): send_msg(self.sock, {'type': 'chat', 'data': text})
    def send_label(self, tet_id, label): send_msg(self.sock, {'type': 'set_label', 'id': tet_id, 'label': label})
    def stop(self):
        self.running = False
        try: self.sock.shutdown(socket.SHUT_RDWR); self.sock.close()
        except OSError: pass

def main():
    print("CLI Options:\n  -connect <ip>:<port>\n  -listen <port>\n  -file <filename>\n  -t <scale> -z <zoom> -o <x,y,z>\n  blacklist.cfg\nTET~CRAFT Initializing...")
    cli_connect_addr, cli_listen_port, cli_load_file = None, None, None
    cli_time_scale, cli_zoom_factor, cli_cam_pan = None, None, None
    args = sys.argv[1:]; i = 0
    while i < len(args):
        if args[i] == '-connect' and i + 1 < len(args): cli_connect_addr = args[i+1]; i += 1
        elif args[i] == '-listen':
            cli_listen_port = DEFAULT_PORT
            if i + 1 < len(args) and args[i+1].isdigit(): cli_listen_port = int(args[i+1]); i += 1
        elif args[i] == '-file' and i + 1 < len(args): cli_load_file = args[i+1]; i += 1
        elif args[i] == '-t' and i + 1 < len(args):
            try: cli_time_scale = float(args[i+1]); i += 1
            except ValueError: print(f"Invalid value for -t: {args[i+1]}")
        elif args[i] == '-z' and i + 1 < len(args):
            try: cli_zoom_factor = float(args[i+1]); i += 1
            except ValueError: print(f"Invalid value for -z: {args[i+1]}")
        elif args[i] == '-o' and i + 1 < len(args):
            try:
                coords = [float(c) for c in args[i+1].split(',')]
                if len(coords) == 3: cli_cam_pan = coords; i += 1
                else: print(f"Invalid format for -o: must be x,y,z. Got: {args[i+1]}")
            except ValueError: print(f"Invalid coordinates for -o: {args[i+1]}")
        i += 1

    global WIDTH, HEIGHT, clock, game_mode, host_instance, guest_instance, net_avatars, net_messages
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"; pygame.init(); pygame.mixer.init(44100, -16, 2, 512)
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE); pygame.display.set_caption("TET~CRAFT")
    clock = pygame.time.Clock(); font_l = pygame.font.SysFont('Georgia', 32); font_s = pygame.font.SysFont(None, 24)
    world = World(generate_boing_sound()); cam = Camera()
    show_intro(screen, cam); ping_sound = generate_ping_sound()

    flags = {'t0': False, 't1': False, 't2': False, 'j1': False, 't3': False}; msgs = []
    dragging, rotating, last_mouse = None, False, (0,0); time_scale = 1.0; reset_timer = None
    locked_sticky_target, sticky_unlock_timer = None, None; frame_count = 0
    rmb_down_timer, rmb_start_pos = None, None; past_projection = PastProjection4Sphere()
    animation_state = 'IDLE'
    disk_surf = None # REUSE SURFACE OPTIMIZATION

    if cli_time_scale is not None: time_scale = cli_time_scale
    if cli_zoom_factor is not None: cam.dist = DEFAULT_CAM_DIST / max(0.01, cli_zoom_factor)
    if cli_cam_pan is not None: cam.pan = (np.array(cli_cam_pan) - 0.5) * 2.0 * FIELD_SCALE

    def add_timed_message(text, y_offset=0, duration=4): msgs.append([text, y_offset, pygame.time.get_ticks() + duration * 1000])
    def add_network_message(text): net_messages.append([text, time.time() + 8])
    def reset_simulation(show_message=True):
        nonlocal flags, time_scale, world, cam, dragging, rotating, locked_sticky_target, sticky_unlock_timer, animation_state
        global game_mode, host_instance, guest_instance, net_avatars, net_messages
        if host_instance: host_instance.stop(); host_instance = None
        if guest_instance: guest_instance.stop(); guest_instance = None
        world.tets.clear(); world.joints.clear(); world.sticky_pairs.clear(); past_projection.points.clear()
        net_avatars.clear(); net_messages.clear(); flags = {k: False for k in flags}
        cam.__init__(); time_scale = 1.0; game_mode = 'single_player'; dragging, rotating = None, False
        locked_sticky_target, sticky_unlock_timer = None, None; animation_state = 'IDLE'
        if show_message: add_timed_message("Simulation Reset", duration=2)
    def save_world_to_file():
        if not world.tets: add_timed_message("Cannot save an empty kleinverse.", duration=3); return
        try:
            with open(SAVE_FILENAME, 'w') as f: json.dump(world.get_state(), f, cls=NumpyEncoder, indent=2)
            add_timed_message(f"Saved kleinverse to {SAVE_FILENAME}", duration=3)
        except IOError as e: add_timed_message(f"Error saving file: {e}", duration=4)
    def connect_as_guest(host_ip, port):
        global guest_instance, game_mode
        try:
            save_world_to_file(); guest_instance, game_mode = Guest(host_ip, int(port), world, cam, add_network_message, ping_sound), 'guest'
            add_timed_message(f"Connected to {host_ip}:{port}", duration=3); world.tets.clear(); world.joints.clear(); world.sticky_pairs.clear(); return True
        except Exception as e: add_timed_message(f"Failed to connect: {e}", duration=4); return False
    def discover_and_join():
        global guest_instance, game_mode
        if game_mode != 'single_player': return
        host_addr = None
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1); sock.settimeout(1.0)
                sock.sendto(b"DISCOVER_TETCRAFT_HOST", ('<broadcast>', DISCOVERY_PORT))
                data, addr = sock.recvfrom(1024)
                if data.startswith(b"TETCRAFT_HOST_HERE:"): host_addr = (addr[0], int(data.split(b':')[1]))
        except socket.timeout: pass
        if host_addr: connect_as_guest(host_addr[0], host_addr[1])
        else:
            user_input = get_user_input(screen, "Enter Host IP:Port (e.g., 192.168.1.5:65420):")
            if user_input:
                parts = user_input.split(':'); ip = parts[0]; port = int(parts[1]) if len(parts) > 1 else DEFAULT_PORT
                connect_as_guest(ip, port)
    def initiate_host_mode(port=None):
        global host_instance, game_mode
        if game_mode == 'single_player':
            host_instance, game_mode = Host(world, add_network_message, ping_sound, port), 'host'
            add_timed_message(f"Hosting on port {host_instance.port}", duration=3)

    loaded_from_save = False
    if cli_connect_addr: connect_as_guest(*(cli_connect_addr.split(':') if ':' in cli_connect_addr else (cli_connect_addr, DEFAULT_PORT)))
    elif cli_listen_port: initiate_host_mode(cli_listen_port)
    elif cli_load_file and os.path.exists(cli_load_file):
        try:
            with open(cli_load_file, 'r', encoding='utf-8-sig') as f: world.set_state(json.load(f))
            add_timed_message(f"Loaded {cli_load_file}"); loaded_from_save = True
        except Exception as e: print(f"Error loading '{cli_load_file}': {e}")
    if not (loaded_from_save or cli_connect_addr or cli_listen_port): show_void_screen(screen, world)

    running = True
    while running:
        unscaled_dt = min(0.1, clock.tick(FPS) / 1000.0)
        governor_active = clock.get_fps() < 55 if frame_count > 60 else False
        scaled_dt, frame_count, fps = unscaled_dt * time_scale, frame_count + 1, clock.get_fps()
        if 0 < fps < 45 and time_scale > 1.0: time_scale = max(1.0, time_scale * 0.99)
        is_interactive, hovered_vertex = (game_mode in ['single_player', 'host']), None

        # Build World Vertices ONCE per frame
        # If no tets, empty arrays
        if world.tets:
            all_verts_world = np.array([t.local + t.pos for t in world.tets]).reshape(-1, 3)
            # Pre-project for mouse logic
            all_verts_screen = cam.project_many(all_verts_world)
        else:
            all_verts_world = np.empty((0, 3))
            all_verts_screen = np.empty((0, 2))

        # Mouse Logic using cached arrays
        if is_interactive and not rotating and world.tets:
            mx, my = pygame.mouse.get_pos()
            mouse_arr = np.array([mx, my], dtype=np.float64)

            # Using cached all_verts_screen
            dist_sq = np.sum((all_verts_screen - mouse_arr)**2, axis=1)
            min_dist_sq_val = np.min(dist_sq) if dist_sq.size > 0 else float('inf')

            if min_dist_sq_val < SELECTION_RADIUS**2:
                min_idx = np.argmin(dist_sq)
                # Ensure the point is actually in front of the camera (masked value is -99999)
                if all_verts_screen[min_idx][0] > -9000:
                    hovered_vertex = (world.tets[min_idx // 4], min_idx % 4)

        if is_interactive and hovered_vertex and not dragging:
            h_tet = hovered_vertex[0]
            h_tet.pos_prev[:] = h_tet.pos[:]
            h_tet.local_prev[:] = h_tet.local[:]

        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            if e.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = e.w, e.h
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                disk_surf = None # Reset surface
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_v and is_interactive: save_world_to_file()
                if e.key == pygame.K_BACKQUOTE and not reset_timer:
                    if is_interactive: world.explode()
                    add_timed_message("RESETTING...", duration=2); reset_timer = pygame.time.get_ticks() + 2000
                if is_interactive and e.key == pygame.K_SPACE: (world.spawn() if len(world.tets) < 2 else (world.spawn_polar_pair() if flags['j1'] else None))
                if e.key == pygame.K_x: cam.pan = world.center_of_mass.copy(); add_timed_message("Camera recentered")
                if e.key == pygame.K_h: initiate_host_mode()
                if e.key == pygame.K_TAB: discover_and_join()
                if e.key == pygame.K_t: time_scale = 1.0; add_timed_message("Time scale reset")
            if e.type == pygame.MOUSEBUTTONDOWN:
                clicked_avatar_id = None
                for aid, ad in net_avatars.items():
                    if np.linalg.norm(np.array(e.pos) - cam.project(ad['pos'])) < max(15, 40*DEFAULT_CAM_DIST/(cam.dist+1e-6)): clicked_avatar_id = aid; break
                if clicked_avatar_id:
                    chat_msg = get_user_input(screen, f"Message to {clicked_avatar_id.split('_')[0]}:")
                    if chat_msg:
                        if game_mode == 'guest' and guest_instance: guest_instance.send_chat(chat_msg); add_network_message(f"You to Host: {chat_msg}")
                        elif game_mode == 'host' and host_instance:
                            target_sock = next((s for s,d in host_instance.clients.items() if d['id']==clicked_avatar_id), None)
                            if target_sock: send_msg(target_sock, {'type':'chat','data':chat_msg}); add_network_message(f"You to {clicked_avatar_id.split(':')[0]}: {chat_msg}")
                else:
                    if is_interactive and e.button == 1 and hovered_vertex:
                        start_tet, start_idx = hovered_vertex
                        world.sticky_pairs = [p for p in world.sticky_pairs if not (
                            (p[0].id == start_tet.id and p[1] == start_idx) or
                            (p[2].id == start_tet.id and p[3] == start_idx)
                        )]
                        dragging = (hovered_vertex[0], hovered_vertex[1], cam.get_transformed_z(hovered_vertex[0].verts()[hovered_vertex[1]]))
                        locked_sticky_target, sticky_unlock_timer = None, None
                    if e.button == 3: rotating, last_mouse, rmb_down_timer, rmb_start_pos = True, e.pos, pygame.time.get_ticks(), e.pos
                    if is_interactive and e.button == 1 and world.joints:
                        mouse_pos_arr = np.array(e.pos, dtype=np.float64)
                        cj = min(world.joints, key=lambda j: dist_point_to_line_segment(mouse_pos_arr, np.array(cam.project(j.A.verts()[j.ia]), dtype=np.float64), np.array(cam.project(j.B.verts()[j.ib]), dtype=np.float64)), default=None)
                        if cj:
                            d_check = dist_point_to_line_segment(mouse_pos_arr, np.array(cam.project(cj.A.verts()[cj.ia]), dtype=np.float64), np.array(cam.project(cj.B.verts()[cj.ib]), dtype=np.float64))
                            if d_check < 8: world.joints.remove(cj)

            if e.type == pygame.MOUSEBUTTONUP:
                if is_interactive and e.button == 1 and dragging and locked_sticky_target:
                    t1, i1 = dragging[0], dragging[1]
                    t2, i2 = locked_sticky_target[0], locked_sticky_target[1]
                    current_binds = 0
                    for j in world.joints:
                        if (j.A.id == t1.id and j.ia == i1) or (j.B.id == t1.id and j.ib == i1): current_binds += 1
                    if current_binds < 8:
                        has_desire = False
                        for p in world.sticky_pairs:
                            if (p[0].id == t1.id and p[1] == i1) or (p[2].id == t1.id and p[3] == i1): has_desire = True; break
                        if not has_desire: world.sticky_pairs.append((t1, i1, t2, i2))
                if e.button == 1: dragging, locked_sticky_target, sticky_unlock_timer = None, None, None
                if e.button == 3:
                    if rmb_down_timer and pygame.time.get_ticks() - rmb_down_timer < 300 and np.linalg.norm(np.array(e.pos) - rmb_start_pos) < 5 and world.tets:
                        dist_sq = np.sum((cam.project_many(np.array([t.pos for t in world.tets])) - e.pos)**2, axis=1)
                        if dist_sq.size > 0 and np.min(dist_sq) < (20 * DEFAULT_CAM_DIST / cam.dist)**2:
                            target_tet = world.tets[np.argmin(dist_sq)]; new_label = get_user_input(screen, "Label:", target_tet.label)
                            if game_mode == 'guest' and guest_instance: guest_instance.send_label(target_tet.id, new_label)
                            else: target_tet.label = new_label
                    rotating = False
            if e.type == pygame.MOUSEWHEEL: cam.zoom(ZOOM_SPEED if e.y < 0 else 1/ZOOM_SPEED)

        keys = pygame.key.get_pressed()
        cam.pitch += ORBIT_SPEED * unscaled_dt * (int(keys[pygame.K_w]) - int(keys[pygame.K_s]))
        cam.yaw += ORBIT_SPEED * unscaled_dt * (int(keys[pygame.K_d]) - int(keys[pygame.K_a]))
        if keys[pygame.K_r]: cam.zoom(1/ZOOM_SPEED)
        if keys[pygame.K_f]: cam.zoom(ZOOM_SPEED)
        cam.pan += np.array([math.cos(cam.yaw), 0, -math.sin(cam.yaw)]) * (int(keys[pygame.K_q]) - int(keys[pygame.K_e])) * PAN_SPEED * unscaled_dt * (cam.dist / DEFAULT_CAM_DIST)
        if is_interactive and keys[pygame.K_c] and not governor_active: time_scale = min(10.0, time_scale + 2.0 * unscaled_dt)
        if is_interactive and keys[pygame.K_z]: time_scale = max(0.1, time_scale - 2.0 * unscaled_dt)
        if rotating: mx, my = pygame.mouse.get_pos(); cam.yaw += (mx - last_mouse[0]) * 0.005; cam.pitch = np.clip(cam.pitch - (my - last_mouse[1]) * 0.005, -1.57, 1.57); last_mouse = (mx, my)

        if is_interactive and dragging:
            t_drag, i_drag, dd = dragging; m3d = cam.unproject(pygame.mouse.get_pos(), dd); delta = m3d - t_drag.verts()[i_drag]
            t_drag.local[i_drag] += delta * MOUSE_PULL_STRENGTH * (DEFAULT_CAM_DIST / cam.dist); t_drag.pos += delta * BODY_PULL_STRENGTH * (DEFAULT_CAM_DIST / cam.dist)

            mp2 = np.array(pygame.mouse.get_pos(), dtype=np.float64)
            best_dist_sq = SELECTION_RADIUS**2

            # Use the already calculated screen points
            avs = all_verts_screen
            current_hover_target = None

            for tidx, tt in enumerate(world.tets):
                if tt.id == t_drag.id: continue
                for vidx in range(4):
                    idx_flat = tidx * 4 + vidx
                    pt = avs[idx_flat]
                    d_sq = np.sum((pt - mp2)**2)
                    if d_sq < best_dist_sq:
                        best_dist_sq = d_sq
                        current_hover_target = (tt, vidx)
            locked_sticky_target = current_hover_target
            if locked_sticky_target: sticky_unlock_timer = pygame.time.get_ticks() + 5000

        if is_interactive:
            world.update(scaled_dt, unscaled_dt, time_scale, add_timed_message)
            if game_mode == 'host' and host_instance and frame_count % 2 == 0: host_instance.broadcast_state()
        elif game_mode == 'guest':
            if guest_instance and guest_instance.running:
                if guest_instance.latest_world_state: world.set_state(guest_instance.latest_world_state)
                if frame_count % 5 == 0: guest_instance.send_cam_update()
                if world.tets: cam.pan = world.center_of_mass
            else: reset_simulation(show_message=False); show_void_screen(screen, world)
        if reset_timer and pygame.time.get_ticks() > reset_timer: reset_simulation(); reset_timer = None
        if is_interactive:
            if len(world.tets) == 1 and not flags['t0']: flags['t0'] = True; add_timed_message("LET THERE BE TIME!", -20); add_timed_message("... and let it have a beginning.", 20)
            if len(world.tets) >= 2 and not flags['t2']:
                flags['t2'] = True; add_timed_message("LET THERE BE SEPARATION!", -20); add_timed_message("And let it chase time ever into the future.", 20)
                if len(world.tets) == 2: world.sticky_pairs.extend([(world.tets[0], v, world.tets[1], v) for v in range(4)])
            if len(world.tets) >= 2 and world.joints and not flags['j1']: flags['j1'] = True; add_timed_message("LET THERE BE LIGHT", -20); add_timed_message("To grow old and wise!", 20)
            if len(world.tets) >= 3 and flags['j1'] and not flags['t3']: flags['t3'] = True; add_timed_message("...and, LET THERE BE DARKNESS!", -20); add_timed_message("So light can be misunderstood!", 20)

        # Rendering
        tl = np.clip((time_scale - 0.1) / 9.9, 0, 1)
        bg_color = tuple(np.array([30,0,0]) * (1-tl) + np.array([5,5,10]) * tl if flags['t3'] else ((255,255,255) if flags['j1'] else (10,10,20)))
        screen.fill(bg_color)

        if flags['t3']:
            past_projection.update_and_draw(screen, cam, world.center_of_mass, len(world.tets), time_scale, WIDTH, HEIGHT)

            # Optimized Black Hole Draw
            if disk_surf is None:
                disk_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            disk_surf.fill((0,0,0,0)) # Clear reusable surface
            draw_standard_black_hole_jit(disk_surf, cam, flags, tl, world.center_of_mass)
            screen.blit(disk_surf, (0,0))

        if flags['j1']:
            ac = [(255,0,0), (255,255,255) if flags['t3'] else (0,0,0), (0,255,255)] if flags['t3'] else [(255,0,0), (0,0,0)]
            for i, c in enumerate(ac):
                pv, nv = np.zeros(3), np.zeros(3); pv[i], nv[i] = AXIS_LEN, -AXIS_LEN
                pygame.draw.line(screen, c, cam.project(nv + world.center_of_mass), cam.project(pv + world.center_of_mass), 2)
        elif len(world.tets) > 0: pygame.draw.circle(screen, (255,255,255), cam.project(world.center_of_mass), 3)

        if world.tets:
            # Re-generate arrays for Drawing since Physics moved them
            # This fixes the visual lag/desync
            awv = np.array([t.local + t.pos for t in world.tets]).reshape(-1, 3)
            asv = cam.project_many(awv).reshape(len(world.tets), 4, 2)
            id_idx = {t.id: i for i, t in enumerate(world.tets)}

            if is_interactive and dragging and locked_sticky_target: pygame.draw.line(screen, (255,140,0), asv[id_idx[dragging[0].id], dragging[1]], asv[id_idx[locked_sticky_target[0].id], locked_sticky_target[1]], 2)
            for t1, i1, t2, i2 in world.sticky_pairs:
                if t1.id in id_idx and t2.id in id_idx: pygame.draw.line(screen, (255, 140, 0), asv[id_idx[t1.id], i1], asv[id_idx[t2.id], i2], 1)
            for j in world.joints:
                if j.A.id in id_idx and j.B.id in id_idx: pygame.draw.line(screen, (255, 255, 255), asv[id_idx[j.A.id], j.ia], asv[id_idx[j.B.id], j.ib], 1)

            # Draw Tets
            # Pre-calculate Z-depths for sorting
            z_depths = cam.get_transformed_z_many(np.array([t.pos for t in world.tets]))
            sorted_indices = np.argsort(z_depths)

            for idx in sorted_indices:
                t = world.tets[idx]
                screen_pts = asv[idx]
                world_verts = awv[idx*4:(idx+1)*4]

                cc = list(t.colors) if t.colors else list(Tetrahedron.FACE_COLORS)
                if t.is_magnetized:
                    cc = [(0,0,0)]*4; cc[2 if t.magnetism==1 else 3] = Tetrahedron.FACE_COLORS[2 if t.magnetism==1 else 3]

                # Sort faces
                face_z = np.mean(cam.get_transformed_z_many(world_verts[Tetrahedron.FACES_NP]), axis=1)
                for fidx in np.argsort(face_z)[::-1]:
                    points = screen_pts[Tetrahedron.FACES_NP[fidx]]
                    if not np.any(points < -100): pygame.draw.polygon(screen, cc[fidx], points)

                for i, j in t.EDGES_NP: pygame.draw.line(screen, (0,0,0), screen_pts[i], screen_pts[j], 1)
                vert_color = (0,0,0) if (flags['j1'] and not flags['t3']) else (255,255,255)
                for p in screen_pts: pygame.draw.circle(screen, vert_color, p, 1)
                if t.label: surf = font_s.render(t.label, True, (255,255,0)); screen.blit(surf, surf.get_rect(center=cam.project(t.pos + [0, 8, 0])))

        for avatar_id, avatar_data in net_avatars.items(): draw_player_avatar(screen, cam, np.array(avatar_data['pos']), avatar_data['color'], avatar_id)

        now_ticks = pygame.time.get_ticks(); text_color = (0,0,0) if (flags['j1'] and not flags['t3']) else (200,200,200)
        msgs = [m for m in msgs if now_ticks < m[2]]
        for ts, yo, et in msgs:
            s = font_l.render(ts, True, text_color); s.set_alpha(max(0, min(255, (et - now_ticks) * 255 / 1000)))
            screen.blit(s, s.get_rect(center=(WIDTH//2, HEIGHT//2 + yo)))
        net_messages = deque([m for m in net_messages if time.time() < m[1]], maxlen=5)
        for i, (txt, et) in enumerate(list(net_messages)):
            s = font_s.render(txt, True, (255,200,100)); s.set_alpha(max(0, min(255, (et - time.time()) * 100))); screen.blit(s, (10, 40+i*25))

        screen.blit(font_s.render(f"FPS: {int(fps)}", True, (255, 255, 0)), (10, 10))
        zf = DEFAULT_CAM_DIST / cam.dist; zoom_text = f"{zf:.1f}x" if zf > 10 else f"{zf:.4f}x" if zf < 0.1 else f"{zf:.2f}x"
        mode_text = f"Mode: {game_mode.replace('_',' ').title()} | Tets: {len(world.tets)} | Zoom: {zoom_text} | Z/C Time ({time_scale:.1f}x)"
        if governor_active and time_scale > 1: mode_text += " (GOV)"
        top_leg = font_s.render(mode_text, True, (0,255,255)); screen.blit(top_leg, top_leg.get_rect(center=(WIDTH//2, 20)))
        bot_leg = font_s.render("WASD/RMB View|QE Pan|R/F Zoom|V Save|H Host|TAB Join|~ Reset|T/X Utils", True, (0,255,255))
        screen.blit(bot_leg, bot_leg.get_rect(center=(WIDTH//2, HEIGHT-25)))

        pygame.display.flip()

    if host_instance: host_instance.stop()
    if guest_instance: guest_instance.stop()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
