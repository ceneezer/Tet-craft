PSEUDOCODE="""

TetCraft Program Explained in Pseudo-Code
1. Configuration & Constants

This section defines the global parameters that govern the simulation's behavior.

code
Code
download
content_copy
expand_less
// -- Core Simulation --
EDGE_LEN = 2.0                // The fixed length of a tetrahedron edge
COLLISION_RADIUS = 1.5        // The size of a tetrahedron for collision
DAMPING = 0.995               // Energy loss factor for Verlet integration

// -- Metaphysical / Behavioral Laws (Refactored) --
K_ORIGIN_PULL = 0.00005       // Gentle pull towards the universe's center of mass

// [Critique #3] Cohesion based on Harmony
K_STICKY_PULL = 0.001         // Base strength of the "desire" connection
STICKY_PULL_HARMONY_SENSITIVITY = 0.1 // How sensitive the pull is to energy differences

// [Critique #2] Magnetism as a Memory-Based Influence
K_MAGNETIC_TORQUE = 0.05      // Strength of the rotational "nudge" from magnetism
K_MAGNETIC_BIAS_BUILDUP = 0.1 // How quickly magnetic memory forms
MAGNETIC_BIAS_DECAY = 0.998   // How slowly magnetic memory fades (hysteresis)

// -- Energy and Emergent Containment (Refactored) --
// [Critique #1] Replaced hard walls with a "cold edge"
BATTERY_CHARGE_RATE = 0.03    // Rate of energy gain near the center
BATTERY_DRAIN_RATE = 0.005    // Base rate of energy loss
CRITICAL_RADIUS = 1200        // Distance from origin where energy drain becomes severe
2. Main Data Structures (Classes)

These are the primary objects that make up the simulation world.

code
Code
download
content_copy
expand_less
CLASS Camera:
    PROPERTIES:
        yaw, pitch, distance, pan_vector(x,y,z)
    METHODS:
        project(world_vec): Converts a 3D world coordinate to a 2D screen coordinate.
        unproject(screen_pos, depth): Converts a 2D screen coordinate back to 3D.
        zoom(factor): Adjusts the camera distance.

CLASS Tetrahedron:
    PROPERTIES:
        id
        position(x,y,z), previous_position(x,y,z)       // For Verlet physics
        local_vertices[4], previous_local_vertices[4]   // Stores shape/rotation
        battery (0.0 to 1.0)                            // Represents energy or "heat"
        is_magnetized (boolean)
        magnetism (polarity: +1 or -1)
        orientation_bias(x,y,z)                         // [Critique #2] Vector storing magnetic memory

CLASS World:
    PROPERTIES:
        list_of_tetrahedrons
        list_of_joints
        list_of_sticky_pairs (desires)
        center_of_mass_vector
    METHODS:
        spawn(): Creates a new tetrahedron.
        update(dt): Runs one full physics and logic tick for the entire simulation.
        get_state(): Serializes the entire world state into a dictionary for saving/networking.
        set_state(state_dict): Reconstructs the world from a saved state.
3. Core Physics Engine (JIT-Optimized Functions)

These are the high-performance Numba functions that handle all the heavy calculations. They operate on large NumPy arrays for maximum speed.

code
Code
download
content_copy
expand_less
FUNCTION world_update_physics_jit(all_positions, all_batteries, ...):
    // This is the heart of the "Law of Universal Balance"

    1. // -- Gravity & Cohesion --
       Initialize an `acceleration` array for all tets to zeros.
       Apply a gentle `K_ORIGIN_PULL` towards the center of mass for all tets.

    2. // [Critique #3] Apply Desire Forces (Cohesion Law)
       FOR each sticky_pair (desire):
           Calculate the energy difference between the two connected tets.
           Calculate a `harmony_factor` = 1 / (energy_difference + sensitivity).
           The force of the desire is `K_STICKY_PULL * harmony_factor`.
           Apply this force to pull the two tets together.

    3. // [Critique #1] Apply Energy Dynamics & Emergent Containment
       Calculate each tet's distance from the origin.
       IF a tet is past the `CRITICAL_RADIUS`:
           Calculate a `drain_multiplier` that increases sharply with distance.
           Apply a severe energy drain: `battery -= drain_rate * drain_multiplier`.
       ELSE:
           Apply normal energy charge/drain rates.
       Clip all battery values between 0 and 1.

    4. // [Critique #1] Apply Battery Force & Emergent Orbital Stability
       Calculate the implicit tangential velocity for each tet.
       Calculate a `repulsion_damp_factor` based on this velocity (faster = more dampening).
       Calculate the base battery force (attraction if cold, repulsion if hot).
       IF the force is repulsive:
           Multiply it by the `repulsion_damp_factor`.
       Apply the final (potentially dampened) battery force.

    5. // -- Verlet Integration --
       Update all `positions` based on `previous_positions` and accumulated `accelerations`.

    6. // -- Internal Shape Constraint --
       Iterate 3 times:
           FOR each tetrahedron:
               Check the length of all 6 internal edges.
               If an edge is too long or too short, slightly move its two vertices to correct the length.
       RETURN updated positions, batteries, etc.

FUNCTION update_magnetic_effects_jit(all_locals, all_biases, ...):
    // This function handles the "Law of Magnetic Constraint" as a memory-based influence.

    1. // [Critique #2] Update Magnetic Memory (Hysteresis)
       FOR each magnetized tet (`tet_A`):
           Calculate the net magnetic field vector at its position from all OTHER magnets.
           Nudge `tet_A`'s `orientation_bias` vector towards this net field.
           Apply a slow decay to the `orientation_bias` (`bias *= MAGNETIC_BIAS_DECAY`).

    2. // [Critique #2] Apply Torque
       Calculate the magnetic moment of `tet_A` (its internal magnetic direction).
       Calculate the `torque` required to align the moment with the (net_field + orientation_bias).
       Apply this `torque` as a small, cumulative rotation to `tet_A`'s local vertices.

    RETURN updated local vertices and orientation biases.

FUNCTION conserve_momentum_jit(...)
    Calculate the total momentum of the entire system.
    Subtract the average momentum from every tetrahedron to prevent the whole system from drifting.

FUNCTION resolve_collisions_jit(...)
    Use a k-d tree to find all pairs of tets that are too close.
    For each overlapping pair, push them directly apart.
4. Networking Subsystem

Handles multiplayer functionality, clearly separating the "host" (who runs the simulation) from "guests" (who just receive updates).

code
Code
download
content_copy
expand_less
CLASS Host:
    Starts a server and listens for connecting guests.
    Maintains a list of connected clients.
    Periodically calls `world.get_state()` and broadcasts the result to all guests.
    Receives camera position updates from guests to display their avatars.

CLASS Guest:
    Connects to a host's IP address.
    Runs a listener thread in the background.
    WHEN a 'world_state' message is received:
        It overwrites its local world by calling `world.set_state(received_state)`.
    Periodically sends its own camera position to the host.
5. Main Application Logic (The main function)

This is the high-level conductor that orchestrates initialization, the game loop, input, and rendering.

code
Code
download
content_copy
expand_less
PROCEDURE main():
    // -- 1. Initialization --
    Initialize Pygame, the screen, and the clock.
    Create a `World` object and a `Camera` object.
    Start a background thread to "prime" (pre-compile) all JIT functions while showing an intro screen.
    Check for command-line arguments to start as a host, connect as a guest, or load a file.
    IF no arguments are given:
        Attempt to load `tetcraft_save.json`.
        IF no save file, show a "void screen" until the user presses SPACE to spawn the first tet.

    // -- 2. Main Game Loop --
    LOOP forever:
        // -- A. Timekeeping --
        Calculate `delta_time` since the last frame.
        Update the simulation `time_scale` based on key presses (Z/C keys) or animations.
        `scaled_dt = delta_time * time_scale`.

        // -- B. Input Handling --
        Process all Pygame events (quit, resize, keyboard, mouse).
        Handle camera movement (WASD, QE, mouse drag).
        Handle simulation controls (SPACE to spawn, `~` to reset, V to save).
        Handle networking controls (H to host, TAB to join).

        // -- C. Physics & State Update --
        IF game_mode is 'single_player' OR 'host':
            // We are in charge of the simulation
            CALL `world.update(scaled_dt)`.
            IF game_mode is 'host':
                Broadcast the world state to all guests.
        ELSE IF game_mode is 'guest':
            // We are just a viewer, so we load the state received from the network
            IF a new state has arrived from the host:
                CALL `world.set_state(latest_received_state)`.
            Send our camera position back to the host.

        // -- D. Rendering --
        Clear the screen with the appropriate background color (changes with game progress).
        Draw decorative elements (axes, background particle field).
        Draw lines for all joints (white) and desires (orange).
        Get a list of all tetrahedrons and sort them by their distance from the camera (depth sorting).
        FOR each tetrahedron IN the sorted list:
            Determine its color based on its state (normal, magnetized).
            Sort its 4 faces by their distance from the camera.
            Draw each face as a filled polygon.
            Draw the edges of the tetrahedron.
        Draw all UI overlays (FPS, info text, help text, chat messages).

        // -- E. Display --
        Flip the Pygame display to show the newly drawn frame.
    END LOOP

    // -- 3. Cleanup --
    Stop any running network threads.
    Quit Pygame.
END PROCEDURE
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
import select
import time

# +++ OPTIMIZATION IMPORTS +++
from numba import njit
from scipy.spatial import cKDTree

# ============================
# CONFIG
# ============================
WIDTH, HEIGHT = 1200, 800
FPS = 60

EDGE_LEN = 2.0
SNAP_DIST = 0.75
AXIS_LEN = 20

# --- REFACTORED COHESION CONSTANTS ---
K_STICKY_PULL = 0.001
STICKY_PULL_HARMONY_SENSITIVITY = 0.1
STICKY_EXPONENTIAL_THRESHOLD = 100.0 * EDGE_LEN
NEIGHBOR_DESIRE_THRESHOLD = 1000.0 * EDGE_LEN

DEFAULT_PORT = 65420
PORT_RANGE = range(DEFAULT_PORT, DEFAULT_PORT + 10)
DISCOVERY_PORT = 65419

# --- VERLET PHYSICS CONSTANTS ---
DAMPING = 0.995
MOUSE_PULL_STRENGTH = 0.0005
BODY_PULL_STRENGTH = 0.0008
COLLISION_RADIUS = EDGE_LEN * 0.75

# --- REFACTORED MAGNETISM CONSTANTS ---
K_MAGNETIC_TORQUE = 0.05
K_MAGNETIC_BIAS_BUILDUP = 0.1
MAGNETIC_BIAS_DECAY = 0.998
MAGNETIC_EPSILON_SQ = 0.1

# --- CAMERA CONTROL CONSTANTS ---
ORBIT_SPEED = 1.5
PAN_SPEED = 200.0
ZOOM_SPEED = 1.05
FOCAL_LENGTH = 650.0
DEFAULT_CAM_DIST = 70.0
MIN_ZOOM_DIST = DEFAULT_CAM_DIST / 100.0
MAX_ZOOM_DIST = DEFAULT_CAM_DIST / 0.01

# =========================================================================
# v3.0 UNIFIED LAW OF BALANCE CONSTANTS
# =========================================================================
K_UNIFIED_FORCE = 0.0000002

# --- AMBIENT ENERGY FIELD PARAMETERS ---
FIELD_AMPLITUDE = 1.2
FIELD_SCALE = 250.0
FIELD_LINEAR_DECAY = 0.0001
FIELD_QUADRATIC_DECAY = 0.000001

# --- ENERGY EQUILIBRIUM PARAMETERS ---
ENERGY_EQUILIBRIUM_RATE = 0.05
# =========================================================================

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
    depth[np.abs(depth) < 0.1] = 0.1
    scale = FOCAL_LENGTH / depth
    screen_x = width / 2 + x_rot * scale
    screen_y = height / 2 - y_rot * scale
    screen_x[depth <= 0.1] = -1000
    screen_y[depth <= 0.1] = -1000
    return np.stack((screen_x, screen_y), axis=1)

@njit(fastmath=True, cache=True)
def get_transformed_z_many_jit(vecs, pan, yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    transformed = vecs - pan
    x, y, z = transformed[:, 0], transformed[:, 1], transformed[:, 2]
    zz = sy * x + cy * z
    zz2 = sp * y + cp * zz
    return zz2

@njit(fastmath=True, cache=True)
def world_update_physics_jit(
    positions, positions_prev, locals, locals_prev, batteries,
    scaled_dt, time_scale, edges, sticky_pairs_data
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

# ============================
# CAMERA & CORE CLASSES
# ============================
class Camera:
    def __init__(self):
        self.yaw, self.pitch, self.dist, self.pan = 0.0, 0.35, DEFAULT_CAM_DIST, np.zeros(3)
    def get_transformed_z(self, v):
        v = v - self.pan; cy, sy = math.cos(self.yaw), math.sin(self.yaw); cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x, y, z = v; zz = sy*x + cy*z; zz2 = sp*y + cp*zz
        return zz2
    def project(self, v):
        global WIDTH, HEIGHT
        v = v - self.pan; cy, sy = math.cos(self.yaw), math.sin(self.yaw); cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x, y, z = v; x, z = cy*x - sy*z, sy*x + cy*z; y, z = cp*y - sp*z, sp*y + cp*z
        depth = self.dist + z
        if depth <= 0.1: return (-1000, -1000)
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
@njit(cache=True)
def dist_point_to_line_segment(p, a, b):
    ap = p - a; ab = b - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-9)
    t = max(0, min(1, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def generate_boing_sound():
    mixer_settings = pygame.mixer.get_init()
    if mixer_settings is None: return None
    sample_rate, _, channels = mixer_settings; duration = 0.2; num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, False); freq = np.linspace(660.0, 220.0, num_samples); wave = np.sin(2 * np.pi * freq * t) * np.exp(-t * 10)
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
        self.battery = random.uniform(0.3, 0.6)
        self.orientation_bias = np.zeros(3, dtype=np.float64)
        self.colors = None; self.label = ""; self.id = id(self)
        self.is_magnetized = False; self.magnetism = 0
    def verts(self): return self.local + self.pos

class PastTetrahedron(Tetrahedron):
    def __init__(self, pos):
        super().__init__(pos)
        self.colors = [(0,0,0)] * 4

class PastClump:
    def __init__(self):
        self.tets = []
        self.yaw, self.pitch, self.roll = 0, 0, 0
        self.pos = np.array([0, 0, -2500])
    def update(self, num_tets_in_world, time_scale, center_of_mass):
        self.pos = center_of_mass + np.array([0, 0, -2500])
        target_num_tets = max(1, num_tets_in_world - 3)
        while len(self.tets) < target_num_tets: self.tets.append(PastTetrahedron(np.random.uniform(-10, 10, 3)))
        while len(self.tets) > target_num_tets: self.tets.pop()
        spin_factor = (time_scale - 1.0) * 0.5
        if time_scale < 0.5: spin_factor += (0.5 - time_scale) * 2.0
        if time_scale > 7.5: spin_factor += (time_scale - 7.5) * 0.5
        self.yaw += 0.002 * spin_factor; self.pitch += 0.003 * spin_factor; self.roll += 0.005 * spin_factor
        cy, sy = math.cos(self.yaw), math.sin(self.yaw); cp, sp = math.cos(self.pitch), math.sin(self.pitch); cr, sr = math.cos(self.roll), math.sin(self.roll)
        rot_matrix = np.array([[cp*cy, sp*sr - cp*sy*cr, sp*cr + cp*sy*sr], [sy, cy*cr, -cy*sr], [-sp*cy, cp*sr + sp*sy*cr, cp*cr - sp*sy*sr]])
        color_lerp = np.clip((time_scale - 0.5) / 7.0, 0, 1)
        color = np.array([180, 50, 50]) * color_lerp + np.array([50, 50, 180]) * (1 - color_lerp)
        for i, tet in enumerate(self.tets):
            offset = np.array([math.sin(i*2.1), math.cos(i*1.7), math.sin(i*0.8)]) * (len(self.tets) * 3)
            tet.pos = self.pos + np.dot(rot_matrix, offset)
            tet.local = np.dot(tet.REST_NP, rot_matrix.T)
            tet.colors = [np.clip(color, 0, 255)] * 4

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
        self.tets.append(new_tet)
    def spawn_polar_pair(self):
        if not self.tets: return
        parent_tet = random.choice(self.tets); pos1 = parent_tet.pos + norm(np.random.rand(3)) * EDGE_LEN * 5
        pos2 = pos1 + norm(np.random.rand(3)) * EDGE_LEN * 3; tet1, tet2 = Tetrahedron(pos1), Tetrahedron(pos2)
        self.tets.extend([tet1, tet2]); polar_face_idx = random.choice([2, 3]); face_verts = Tetrahedron.FACES_NP[polar_face_idx]
        for i in range(3): self.sticky_pairs.append((tet1, face_verts[i], tet2, face_verts[i]))
    def check_magnetization(self):
        tet_map = {t.id: t for t in self.tets}
        for t in self.tets: t.is_magnetized, t.magnetism = False, 0
        for t in self.tets:
            joints_by_partner = {}
            for j in self.joints:
                partner_id, my_idx = (j.B.id, j.ia) if j.A.id == t.id else ((j.A.id, j.ib) if j.B.id == t.id else (None, None))
                if partner_id:
                    if partner_id not in joints_by_partner: joints_by_partner[partner_id] = set()
                    joints_by_partner[partner_id].add(my_idx)
            for partner_id, connected_indices in joints_by_partner.items():
                for face_idx, polarity in Tetrahedron.FACE_POLARITY_MAP.items():
                    face_verts = set(Tetrahedron.FACES_NP[face_idx])
                    if connected_indices.issuperset(face_verts):
                        t.is_magnetized, t.magnetism = True, polarity
                        partner_tet = tet_map.get(partner_id)
                        if partner_tet: partner_tet.is_magnetized, partner_tet.magnetism = True, polarity
                        break
                if t.is_magnetized: break
    def explode(self):
        self.joints.clear(); self.sticky_pairs.clear()
        for t in self.tets: t.pos_prev = t.pos - np.random.uniform(-1, 1, 3); t.local_prev = t.local - np.random.uniform(-0.5, 0.5, (4,3))
    def try_snap(self, A, ia, B, ib):
        for j in self.joints:
            if (j.A.id, j.ia, j.B.id, j.ib) in [(A.id,ia,B.id,ib), (B.id,ib,A.id,ia)]: return
        self.joints.append(VertexJoint(A, ia, B, ib))
        if self.sound: self.sound.play()
    def calculate_dynamic_center(self):
        if not self.tets: return np.zeros(3)
        return np.mean(np.array([t.pos for t in self.tets]), axis=0)

    def update(self, scaled_dt, unscaled_dt, time_scale, add_msg_fn):
        if not self.tets: return
        self.check_magnetization()
        self.center_of_mass = self.calculate_dynamic_center()

        positions, positions_prev = np.array([t.pos for t in self.tets]), np.array([t.pos_prev for t in self.tets])
        locals_arr, locals_prev = np.array([t.local for t in self.tets]), np.array([t.local_prev for t in self.tets])
        batteries, id_to_idx = np.array([t.battery for t in self.tets]), {t.id: i for i, t in enumerate(self.tets)}
        orientation_biases = np.array([t.orientation_bias for t in self.tets])
        sticky_pairs_data = np.array([[id_to_idx[p[0].id], p[1], id_to_idx[p[2].id], p[3]] for p in self.sticky_pairs if p[0].id in id_to_idx and p[2].id in id_to_idx], dtype=np.int32) if self.sticky_pairs else np.empty((0, 4), dtype=np.int32)

        positions, positions_prev, locals_arr, locals_prev, batteries = world_update_physics_jit(
            positions, positions_prev, locals_arr, locals_prev, batteries,
            scaled_dt, time_scale, Tetrahedron.EDGES_NP, sticky_pairs_data
        )

        magnet_indices = np.array([i for i, t in enumerate(self.tets) if t.is_magnetized], dtype=np.int32)
        magnet_polarities = np.array([t.magnetism for t in self.tets if t.is_magnetized], dtype=np.float64)
        locals_arr, orientation_biases = update_magnetic_effects_jit(locals_arr, orientation_biases, positions, magnet_indices, magnet_polarities, scaled_dt)

        positions_prev = conserve_momentum_jit(positions, positions_prev)

        tree = cKDTree(positions)
        if len(self.tets) >= 2:
            distances, indices = tree.query(positions, k=2); stray_indices = np.where(distances[:, 1] > NEIGHBOR_DESIRE_THRESHOLD)[0]
            if stray_indices.size > 0:
                existing_connections = set(tuple(sorted((j.A.id, j.B.id))) for j in self.joints)
                existing_connections.update(set(tuple(sorted((p[0].id, p[2].id))) for p in self.sticky_pairs))
                for idx in stray_indices:
                    stray_tet, neighbor_tet = self.tets[idx], self.tets[indices[idx, 1]]
                    if tuple(sorted((stray_tet.id, neighbor_tet.id))) in existing_connections: continue
                    self.sticky_pairs.append((stray_tet, random.randint(0, 3), neighbor_tet, random.randint(0, 3)))
                    add_msg_fn("Forced Desire to prevent drifting", duration=2); break

        pairs = tree.query_pairs(r=COLLISION_RADIUS * 2)
        if pairs: positions = resolve_collisions_jit(positions, np.array(list(pairs)))

        if self.joints:
            joint_data = [[id_to_idx.get(j.A.id), j.ia, id_to_idx.get(j.B.id), j.ib] for j in self.joints if id_to_idx.get(j.A.id) is not None and id_to_idx.get(j.B.id) is not None]
            if joint_data:
                joint_data_np = np.array(joint_data, dtype=np.int32)
                for _ in range(3): locals_arr = resolve_joints_jit(locals_arr, joint_data_np)

        for pair in self.sticky_pairs[:]:
            t1, i1, t2, i2 = pair; idx1, idx2 = id_to_idx.get(t1.id), id_to_idx.get(t2.id)
            if idx1 is None or idx2 is None: continue
            p1, p2 = locals_arr[idx1, i1] + positions[idx1], locals_arr[idx2, i2] + positions[idx2]
            if np.linalg.norm(p2 - p1) < SNAP_DIST: self.try_snap(t1, i1, t2, i2); self.sticky_pairs.remove(pair)

        for i, t in enumerate(self.tets):
            t.pos, t.pos_prev, t.local, t.local_prev, t.battery, t.orientation_bias = positions[i], positions_prev[i], locals_arr[i], locals_prev[i], batteries[i], orientation_biases[i]

    def get_state(self):
        tet_states = [{'id': t.id, 'pos': t.pos, 'pos_prev': t.pos_prev, 'local': t.local, 'local_prev': t.local_prev, 'battery': t.battery, 'colors': t.colors, 'label': t.label, 'orientation_bias': t.orientation_bias} for t in self.tets]
        joint_states = [{'A_id': j.A.id, 'ia': j.ia, 'B_id': j.B.id, 'ib': j.ib} for j in self.joints]
        return {'tets': tet_states, 'joints': joint_states}
    def set_state(self, state):
        self.tets.clear(); self.joints.clear(); self.sticky_pairs.clear(); tet_map = {}
        for ts in state['tets']:
            t = Tetrahedron(ts['pos'])
            t.id = ts['id']
            t.pos_prev = np.array(ts['pos_prev'])
            t.local = np.array(ts['local'])
            t.local_prev = np.array(ts['local_prev'])
            t.battery = ts['battery']
            t.colors = ts['colors']
            t.label = ts.get('label', "")
            # Safely load orientation_bias, providing a default for older save files
            t.orientation_bias = np.array(ts.get('orientation_bias', [0.0, 0.0, 0.0]))
            self.tets.append(t)
            tet_map[t.id] = t
        for js in state['joints']:
            if js['A_id'] in tet_map and js['B_id'] in tet_map:
                self.joints.append(VertexJoint(tet_map[js['A_id']], js['ia'], tet_map[js['B_id']], js['ib']))

net_avatars = {}; net_messages = deque(maxlen=5); game_mode = 'single_player'; host_instance, guest_instance = None, None

def prime_jit_functions(cam):
    num_dummy = 4; dummy_pos = np.random.rand(num_dummy, 3) * 100; dummy_pos_prev = dummy_pos.copy()
    dummy_locals = np.random.rand(num_dummy, 4, 3); dummy_locals_prev = dummy_locals.copy(); dummy_batteries = np.random.rand(num_dummy)
    dummy_edges, dummy_sticky_pairs = Tetrahedron.EDGES_NP, np.array([[0, 0, 1, 1], [2, 3, 1, 0]], dtype=np.int32)
    dummy_magnet_indices, dummy_magnet_polarities, dummy_orientation_biases, dummy_joints = np.array([0, 1], dtype=np.int32), np.array([1.0, -1.0], dtype=np.float64), np.zeros((num_dummy, 3)), np.array([[0, 2, 3, 1]], dtype=np.int32)
    _ = norm_njit(np.array([1.0, 2.0, 3.0])); _ = norm_axis_njit(dummy_pos); _ = project_many_jit(dummy_pos, cam.pan, cam.yaw, cam.pitch, cam.dist, WIDTH, HEIGHT); _ = get_transformed_z_many_jit(dummy_pos, cam.pan, cam.yaw, cam.pitch)
    _ = world_update_physics_jit(dummy_pos, dummy_pos_prev, dummy_locals, dummy_locals_prev, dummy_batteries, 1/60.0, 1.0, dummy_edges, dummy_sticky_pairs)
    _ = update_magnetic_effects_jit(dummy_locals, dummy_orientation_biases, dummy_pos, dummy_magnet_indices, dummy_magnet_polarities, 1/60.0)
    _ = conserve_momentum_jit(dummy_pos, dummy_pos_prev); _ = resolve_collisions_jit(dummy_pos, np.array([[0, 1], [2, 3]])); _ = resolve_joints_jit(dummy_locals, dummy_joints)

def show_intro(screen, cam):
    font_lg, font_sm, font_jit = pygame.font.SysFont('Arial Black', min(WIDTH, HEIGHT)//8), pygame.font.SysFont('Arial', min(WIDTH, HEIGHT)//25), pygame.font.SysFont('Monospace', 18)
    threading.Thread(target=prime_jit_functions, args=(cam,), daemon=True).start()
    while threading.active_count() > 1:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); sys.exit()
        screen.fill((10,10,20)); title, sub = font_lg.render("TET~CRAFT", True, (255, 50, 50)), font_sm.render("A Klein-a-verse of your own!", True, (255, 255, 255))
        screen.blit(title, title.get_rect(center=(WIDTH//2, HEIGHT//2-50))); screen.blit(sub, sub.get_rect(center=(WIDTH//2, HEIGHT//2+50))); jit_surf = font_jit.render("Confining Universe...", True, (0, 255, 0)); screen.blit(jit_surf, (10, 10))
        if pygame.time.get_ticks() % 1000 < 500: screen.blit(font_jit.render("_", True, (0, 255, 0)), (10 + jit_surf.get_width(), 10))
        pygame.display.flip(); clock.tick(30)

def show_void_screen(screen, world):
    font_lg, font_sm = pygame.font.SysFont('Georgia', 40), pygame.font.SysFont('Arial', 24); line1, line2 = font_lg.render("Welcome to the void of (mis)understanding...", True, (200,200,200)), font_sm.render("(Press SPACE to begin)", True, (150,150,150))
    waiting = True
    while waiting:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE: world.spawn(); waiting = False
        screen.fill((10,10,20)); screen.blit(line1, line1.get_rect(center=(WIDTH//2, HEIGHT//2-30))); screen.blit(line2, line2.get_rect(center=(WIDTH//2, HEIGHT//2+30))); pygame.display.flip(); clock.tick(15)

def draw_avatar(screen, cam, pos, color):
    r, a = EDGE_LEN*math.sqrt(3/8), EDGE_LEN/math.sqrt(3); base_verts = np.array([[0,0,r], [EDGE_LEN/2,-a/2,-r/3], [-EDGE_LEN/2,-a/2,-r/3], [0,a,-r/3]])
    verts1, verts2 = base_verts + pos, np.array([v * [1,-1,1] for v in base_verts]) + pos; all_verts = np.vstack([verts1, np.array([verts2[0], verts2[2], verts2[1], verts2[3]])]); screen_pts_list, faces = [cam.project(v) for v in all_verts], [(0,1,3), (0,2,1), (0,3,2), (1,2,3), (4,5,7), (4,6,5), (4,7,6), (5,6,7)]
    for face_indices in sorted(faces, key=lambda face: sum(cam.get_transformed_z(all_verts[i]) for i in face), reverse=True):
        face_pts = [screen_pts_list[i] for i in face_indices]
        try: pygame.draw.polygon(screen, color, face_pts); pygame.draw.aalines(screen, (20,20,20), True, face_pts)
        except: pass

def get_user_input(screen, prompt, initial_text=""):
    input_text, font, active = initial_text, pygame.font.SysFont(None, 32), True
    while active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN: active = False
                elif event.key == pygame.K_BACKSPACE: input_text = input_text[:-1]
                else: input_text += event.unicode
        screen.fill((10,10,20)); prompt_surf, input_surf = font.render(prompt, True, (255,255,255)), font.render(input_text, True, (255,255,0)); screen.blit(prompt_surf, (WIDTH//2 - prompt_surf.get_width()//2, HEIGHT//2 - 50)); screen.blit(input_surf, (WIDTH//2 - input_surf.get_width()//2, HEIGHT//2)); pygame.display.flip(); clock.tick(30)
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
    def __init__(self, world, add_msg_fn, port=None):
        self.world, self.add_msg, self.clients, self.lock, self.server, self.port, self.running = world, add_msg_fn, {}, threading.Lock(), socket.socket(socket.AF_INET, socket.SOCK_STREAM), 0, True
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
                client_sock, addr = self.server.accept(); client_id = f"guest_{addr[0]}:{addr[1]}"
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
            elif msg['type'] == 'chat': self.add_msg(f"<{client_id.split(':')[0]}>: {msg['data']}")
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
        self.running = False
        with self.lock:
            for sock in self.clients: sock.close()
        self.server.close()

class Guest:
    def __init__(self, host_ip, port, world, cam, add_msg_fn):
        self.world, self.cam, self.add_msg, self.sock, self.host_id, self.running, self.latest_world_state = world, cam, add_msg_fn, socket.socket(socket.AF_INET, socket.SOCK_STREAM), f"host_{host_ip}:{port}", True, None
        self.sock.connect((host_ip, port)); threading.Thread(target=self.listen, daemon=True).start()
    def listen(self):
        global net_avatars
        while self.running:
            msg = recv_msg(self.sock)
            if msg is None:
                self.add_msg("Disconnected from host.")
                self.running = False
                break
            if msg['type'] == 'world_state':
                self.latest_world_state = msg['data']['world']
                net_avatars = {self.host_id: {'pos': self.latest_world_state.get('center_of_mass', [0,0,0]), 'color': (50, 50, 255)}}
                if 'avatars' in msg['data']:
                    for av_id, av_data in msg['data']['avatars'].items():
                        if av_id != 'host':
                            net_avatars[av_id] = av_data
            elif msg['type'] == 'chat': self.add_msg(f"<HOST>: {msg['data']}")
    def send_cam_update(self): send_msg(self.sock, {'type': 'cam_update', 'data': self.cam.get_state()})
    def send_chat(self, text): send_msg(self.sock, {'type': 'chat', 'data': text})
    def send_label(self, tet_id, label): send_msg(self.sock, {'type': 'set_label', 'id': tet_id, 'label': label})
    def stop(self):
        self.running = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR); self.sock.close()
        except OSError: pass

def main():
    print("CLI Options:")
    print("  -connect <ip>:<port>   (Connect directly to a server)")
    print("  -listen <port>         (Launch directly as a host on a port)")
    print("  -file <filename>       (Load a specific save file on start)")
    print("TET~CRAFT Initializing...")

    cli_connect_addr, cli_listen_port, cli_load_file = None, None, None
    args = sys.argv[1:]; i = 0
    while i < len(args):
        if args[i] == '-connect' and i + 1 < len(args):
            cli_connect_addr = args[i+1]; i += 1
        elif args[i] == '-listen':
            cli_listen_port = DEFAULT_PORT
            if i + 1 < len(args) and args[i+1].isdigit():
                cli_listen_port = int(args[i+1]); i += 1
        elif args[i] == '-file' and i + 1 < len(args):
            cli_load_file = args[i+1]; i += 1
        i += 1

    global WIDTH, HEIGHT, clock, game_mode, host_instance, guest_instance, net_avatars, net_messages
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"; pygame.init(); pygame.mixer.init(44100, -16, 2, 512)
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE); pygame.display.set_caption("TET~CRAFT")
    clock = pygame.time.Clock(); font_l = pygame.font.SysFont('Georgia', 32); font_s = pygame.font.SysFont(None, 24)
    world = World(generate_boing_sound()); cam = Camera(); show_intro(screen, cam)

    flags = {'t0': False, 't1': False, 't2': False, 'j1': False, 't3': False}; msgs = []
    dragging, rotating, last_mouse = None, False, (0,0); time_scale = 1.0; reset_timer = None
    locked_sticky_target, sticky_unlock_timer = None, None; frame_count = 0
    rmb_down_timer, rmb_start_pos = None, None; past_clump = PastClump(); particles = []
    animation_state = 'IDLE'; animation_start_time, animation_duration = 0, 0
    start_zoom, start_time_scale = 0, 1.0; t2_triggered_animation = False

    def add_timed_message(text, y_offset=0, duration=4):
        msgs.append([text, y_offset, pygame.time.get_ticks() + duration * 1000])
    def add_network_message(text):
        net_messages.append([text, time.time() + 8])

    def reset_simulation(show_message=True):
        nonlocal flags, time_scale, world, cam, dragging, rotating, locked_sticky_target, sticky_unlock_timer, animation_state, t2_triggered_animation
        global game_mode, host_instance, guest_instance, net_avatars, net_messages
        if host_instance: host_instance.stop(); host_instance = None
        if guest_instance: guest_instance.stop(); guest_instance = None
        world.tets.clear(); world.joints.clear(); world.sticky_pairs.clear()
        net_avatars.clear(); net_messages.clear()
        flags = {'t0': False, 't1': False, 't2': False, 'j1': False, 't3': False}
        cam.__init__(); time_scale = 1.0; game_mode = 'single_player'
        dragging, rotating = None, False; locked_sticky_target, sticky_unlock_timer = None, None
        animation_state = 'IDLE'; t2_triggered_animation = False
        if show_message: add_timed_message("Simulation Reset", duration=2)

    def save_world_to_file():
        if world.tets:
            with open(SAVE_FILENAME, 'w') as f: json.dump(world.get_state(), f, cls=NumpyEncoder, indent=2)
            add_timed_message(f"Saved universe to {SAVE_FILENAME}", duration=3)
    def connect_as_guest(host_ip, port):
        global guest_instance, game_mode
        try:
            save_world_to_file(); guest_instance, game_mode = Guest(host_ip, int(port), world, cam, add_network_message), 'guest'
            add_timed_message(f"Connected to {host_ip}:{port}", duration=3)
            world.tets.clear(); world.joints.clear(); world.sticky_pairs.clear(); return True
        except Exception as e:
            add_timed_message(f"Failed to connect: {e}", duration=4); return False
    def discover_and_join():
        global guest_instance, game_mode
        if game_mode != 'single_player': return
        host_addr = None
        for port in PORT_RANGE:
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=0.05): host_addr = ('127.0.0.1', port); break
            except: continue
        if not host_addr:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1); sock.settimeout(2.0)
                try:
                    sock.sendto(b"DISCOVER_TETCRAFT_HOST", ('.'.join(socket.gethostbyname(socket.gethostname()).split('.')[:-1]) + '.255', DISCOVERY_PORT)); data, addr = sock.recvfrom(1024)
                    if data.startswith(b"TETCRAFT_HOST_HERE:"): host_addr = (addr[0], int(data.split(b':')[1]))
                except: pass
        if host_addr: connect_as_guest(host_addr[0], host_addr[1])
        else:
            user_input = get_user_input(screen, "Enter Host IP:Port:")
            if user_input: connect_as_guest(*(user_input.split(':') if ':' in user_input else (user_input, DEFAULT_PORT)))
    def initiate_host_mode(port=None):
        global host_instance, game_mode
        if game_mode == 'single_player':
            host_instance, game_mode = Host(world, add_network_message, port), 'host'
            add_timed_message("Hosting Mode Activated", duration=3)

    loaded_from_save = False
    if cli_connect_addr:
        connect_as_guest(*(cli_connect_addr.split(':') if ':' in cli_connect_addr else (cli_connect_addr, DEFAULT_PORT)))
    elif cli_listen_port:
        initiate_host_mode(cli_listen_port)
    elif cli_load_file:
        if os.path.exists(cli_load_file):
            try:
                with open(cli_load_file, 'r', encoding='utf-8-sig') as f:
                    world.set_state(json.load(f))
                add_timed_message(f"Loaded {cli_load_file}"); loaded_from_save = True
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"Error loading file '{cli_load_file}': {e}. Starting new universe.")
        else:
            print(f"File '{cli_load_file}' not found. Starting new universe.")

    if not (loaded_from_save or cli_connect_addr or cli_listen_port):
        show_void_screen(screen, world)

    running = True
    while running:
        unscaled_dt = min(0.1, clock.tick(FPS) / 1000.0)
        governor_active = clock.get_fps() < 55 if frame_count > 60 else False

        if animation_state != 'IDLE':
            elapsed = pygame.time.get_ticks() - animation_start_time; progress = min(1.0, elapsed / animation_duration)
            if animation_state == 'ZOOMING_OUT':
                cam.dist = start_zoom + (MAX_ZOOM_DIST - start_zoom) * progress
                if progress >= 1.0: animation_state, animation_start_time, animation_duration, start_time_scale = 'SPEEDING_UP', pygame.time.get_ticks(), 30000, time_scale
            elif animation_state == 'SPEEDING_UP':
                time_scale = start_time_scale + (10.0 - start_time_scale) * progress
                if progress >= 1.0: animation_state = 'IDLE'

        scaled_dt, frame_count, fps = unscaled_dt * time_scale, frame_count + 1, clock.get_fps()
        if 0 < fps < 45 and time_scale > 1.0: time_scale = max(1.0, time_scale * 0.99)
        is_interactive, hovered_vertex = (game_mode in ['single_player', 'host']), None

        if is_interactive and not rotating and world.tets:
            mx, my = pygame.mouse.get_pos(); all_verts_screen = cam.project_many(np.array([t.local + t.pos for t in world.tets]).reshape(-1, 3))
            dist_sq = np.sum((all_verts_screen - np.array([mx, my]))**2, axis=1); min_idx = np.argmin(dist_sq)
            if dist_sq[min_idx] < 576: hovered_vertex = (world.tets[min_idx // 4], min_idx % 4)

        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            if e.type == pygame.VIDEORESIZE: WIDTH, HEIGHT = e.w, e.h; screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
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
                if is_interactive and e.button == 1 and hovered_vertex:
                    dragging = (hovered_vertex[0], hovered_vertex[1], cam.get_transformed_z(hovered_vertex[0].verts()[hovered_vertex[1]]))
                    locked_sticky_target, sticky_unlock_timer = None, None
                if e.button == 3: rotating, last_mouse, rmb_down_timer, rmb_start_pos = True, e.pos, pygame.time.get_ticks(), e.pos
                if e.button == 1:
                    clicked_avatar_id = next((aid for aid, ad in net_avatars.items() if np.linalg.norm(np.array(e.pos) - cam.project(ad['pos'])) < 20), None)
                    if clicked_avatar_id:
                        chat_msg = get_user_input(screen, "Message:")
                        if chat_msg:
                            if game_mode == 'guest' and guest_instance: guest_instance.send_chat(chat_msg)
                            elif game_mode == 'host' and host_instance:
                                target_sock = next((s for s, d in host_instance.clients.items() if d['id'] == clicked_avatar_id), None)
                                if target_sock: send_msg(target_sock, {'type': 'chat', 'data': chat_msg})
                    elif is_interactive and world.joints:
                        cj = min(world.joints, key=lambda j: dist_point_to_line_segment(np.array(e.pos, float), np.array(cam.project(j.A.verts()[j.ia]), float), np.array(cam.project(j.B.verts()[j.ib]), float)), default=None)
                        if cj and dist_point_to_line_segment(np.array(e.pos, float), np.array(cam.project(cj.A.verts()[cj.ia]), float), np.array(cam.project(cj.B.verts()[cj.ib]), float)) < 5:
                            world.joints.remove(cj)
            if e.type == pygame.MOUSEBUTTONUP:
                if is_interactive and e.button == 1 and dragging and locked_sticky_target:
                    if dragging[0] != locked_sticky_target[0] or dragging[1] != locked_sticky_target[1]:
                        world.sticky_pairs.append((dragging[0], dragging[1], *locked_sticky_target))
                if e.button == 1: dragging, locked_sticky_target, sticky_unlock_timer = None, None, None
                if e.button == 3:
                    if rmb_down_timer and pygame.time.get_ticks() - rmb_down_timer < 300 and np.linalg.norm(np.array(e.pos) - rmb_start_pos) < 5 and world.tets:
                        tet_positions = np.array([t.pos for t in world.tets])
                        screen_positions = cam.project_many(tet_positions)
                        dist_sq = np.sum((screen_positions - e.pos)**2, axis=1)
                        idx = np.argmin(dist_sq)
                        if dist_sq[idx] < (20 * DEFAULT_CAM_DIST / cam.dist)**2:
                            target_tet = world.tets[idx]
                            new_label = get_user_input(screen, "Label:", target_tet.label)
                            if game_mode == 'guest' and guest_instance: guest_instance.send_label(target_tet.id, new_label)
                            else: target_tet.label = new_label
                    rotating = False
            if e.type == pygame.MOUSEWHEEL: cam.zoom(ZOOM_SPEED if e.y < 0 else 1/ZOOM_SPEED)

        keys = pygame.key.get_pressed()
        cam.pitch += ORBIT_SPEED * unscaled_dt * (int(keys[pygame.K_w]) - int(keys[pygame.K_s]))
        cam.yaw += ORBIT_SPEED * unscaled_dt * (int(keys[pygame.K_d]) - int(keys[pygame.K_a]))
        if keys[pygame.K_r]: cam.zoom(1/ZOOM_SPEED)
        if keys[pygame.K_f]: cam.zoom(ZOOM_SPEED)
        if keys[pygame.K_q] or keys[pygame.K_e]: cam.pan += np.array([math.cos(cam.yaw), 0, -math.sin(cam.yaw)]) * (1 if keys[pygame.K_q] else -1) * PAN_SPEED * unscaled_dt * cam.dist / DEFAULT_CAM_DIST
        if is_interactive and keys[pygame.K_c] and not governor_active and animation_state == 'IDLE': time_scale = min(10.0, time_scale + 2.0 * unscaled_dt)
        if is_interactive and keys[pygame.K_z] and animation_state == 'IDLE': time_scale = max(0.1, time_scale - 2.0 * unscaled_dt)

        if rotating:
            mx, my = pygame.mouse.get_pos(); cam.yaw += (mx - last_mouse[0]) * 0.005; cam.pitch = np.clip(cam.pitch - (my - last_mouse[1]) * 0.005, -1.57, 1.57); last_mouse = (mx, my)

        if is_interactive and dragging:
            t_drag, i_drag, dd = dragging; m3d = cam.unproject(pygame.mouse.get_pos(), dd); delta = m3d - t_drag.verts()[i_drag]
            t_drag.local[i_drag] += delta * MOUSE_PULL_STRENGTH; t_drag.pos += delta * BODY_PULL_STRENGTH
            mp2, best_dist = np.array(pygame.mouse.get_pos(), float), 50
            avs = cam.project_many(np.array([t.local + t.pos for t in world.tets]).reshape(-1, 3))
            current_hover_target = None
            for tidx, tt in enumerate(world.tets):
                for vidx in range(4):
                    if tt == t_drag and vidx == i_drag: continue
                    projected_start_pos = np.array(cam.project(t_drag.verts()[i_drag]), dtype=np.float64)
                    d = dist_point_to_line_segment(avs[tidx*4+vidx], projected_start_pos, mp2)
                    if d < best_dist: best_dist, current_hover_target = d, (tt, vidx)
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
            else:
                reset_simulation(show_message=False)
                show_void_screen(screen, world)

        if reset_timer and pygame.time.get_ticks() > reset_timer: reset_simulation(); reset_timer = None

        if is_interactive:
            if len(world.tets) == 1 and not flags['t0']: flags['t0'] = True; add_timed_message("LET THERE BE TIME!", -20); add_timed_message("... and let it have a beginning.", 20)
            if len(world.tets) >= 2 and not flags['t2']:
                flags['t2'] = True; animation_state, animation_start_time, animation_duration, start_zoom = 'ZOOMING_OUT', pygame.time.get_ticks(), 10000, cam.dist
                add_timed_message("LET THERE BE SEPARATION!", -20); add_timed_message("And let it chase time ever into the future.", 20)
                if len(world.tets) == 2:
                    world.sticky_pairs.extend([(world.tets[0], Tetrahedron.FACES_NP[2,0], world.tets[1], Tetrahedron.FACES_NP[2,0]), (world.tets[0], Tetrahedron.FACES_NP[3,0], world.tets[1], Tetrahedron.FACES_NP[3,0])])
            if len(world.tets) >= 2 and world.joints and not flags['j1']: flags['j1'] = True; add_timed_message("LET THERE BE LIGHT", -20); add_timed_message("To grow old and wise!", 20)
            if len(world.tets) >= 3 and flags['j1'] and not flags['t3']: flags['t3'] = True; add_timed_message("...and, LET THERE BE DARKNESS!", -20); add_timed_message("So light can be seen!", 20)

        tl = np.clip((time_scale - 0.1) / 9.9, 0, 1)
        bg_color = (np.array([30,0,0]) * (1-tl) + np.array([10,10,20]) * tl if flags['t3'] else ((255,255,255) if flags['j1'] else (10,10,20)))
        screen.fill(bg_color)

        if flags['t3']:
            past_clump.update(len(world.tets), time_scale, world.center_of_mass)
            if len(particles) < len(world.tets)*5: particles.extend([np.random.uniform(-1, 1, 3) for _ in range(len(world.tets)*5 - len(particles))])
            pc = np.clip(np.array([120, 100, 100]) + min(len(world.tets), 50), 0, 255)
            for p in particles:
                sp = cam.project(world.center_of_mass + p * FIELD_SCALE * 4)
                if 0 <= sp[0] < WIDTH and 0 <= sp[1] < HEIGHT: screen.set_at(sp, pc)
            cam_forward = np.array([math.sin(cam.yaw)*math.cos(cam.pitch), -math.sin(cam.pitch), math.cos(cam.yaw)*math.cos(cam.pitch)])
            if np.dot(norm(cam.pan - (cam.pan - cam_forward)), [0,0,-1]) > 0.5:
                for ptet in past_clump.tets:
                    try: pygame.draw.polygon(screen, ptet.colors[0], [cam.project(v) for v in ptet.verts()[ptet.FACES_NP[0]]])
                    except: pass

        if flags['j1']:
            ac = [(255,0,0), (255,255,255) if flags['t3'] else (0,0,0)]
            if flags['t3']: ac.append((0,255,255))
            for i, c in enumerate(ac):
                pv, nv = np.zeros(3), np.zeros(3); pv[i], nv[i] = AXIS_LEN, -AXIS_LEN
                pygame.draw.line(screen, c, cam.project(nv + world.center_of_mass), cam.project(pv + world.center_of_mass), 2)
        elif len(world.tets) > 0 and not flags['j1']: pygame.draw.circle(screen, (255,255,255), cam.project(world.center_of_mass), 3)

        if world.tets:
            awv = np.array([t.local + t.pos for t in world.tets]).reshape(-1, 3)
            id_idx = {t.id: i for i, t in enumerate(world.tets)}; asv = cam.project_many(awv).reshape(len(world.tets), 4, 2)
            if is_interactive and dragging and locked_sticky_target:
                pygame.draw.line(screen, (255, 140, 0), asv[id_idx[dragging[0].id], dragging[1]], asv[id_idx[locked_sticky_target[0].id], locked_sticky_target[1]], 2)
            for t1, i1, t2, i2 in world.sticky_pairs:
                if t1.id in id_idx and t2.id in id_idx: pygame.draw.line(screen, (255, 140, 0), asv[id_idx[t1.id], i1], asv[id_idx[t2.id], i2], 1)
            for j in world.joints:
                if j.A.id in id_idx and j.B.id in id_idx: pygame.draw.line(screen, (255, 255, 255), asv[id_idx[j.A.id], j.ia], asv[id_idx[j.B.id], j.ib], 1)

            for idx in np.argsort(cam.get_transformed_z_many(np.array([t.pos for t in world.tets]))):
                t, world_verts, screen_pts = world.tets[idx], awv[idx*4:(idx+1)*4], asv[idx]
                cc = list(t.colors) if t.colors else list(Tetrahedron.FACE_COLORS)
                if t.is_magnetized:
                    polar_idx = 2 if t.magnetism == 1 else 3; cc = [(0,0,0)]*4
                    if t.colors: cc[polar_idx] = Tetrahedron.FACE_COLORS[polar_idx]

                face_depths = np.mean(cam.get_transformed_z_many(world_verts[Tetrahedron.FACES_NP]), axis=1); sorted_face_indices = np.argsort(face_depths)

                if not flags['j1'] and not t.is_magnetized:
                    bc = np.array((200, int(200*t.battery), 255-int(200*t.battery))); am = bc * 0.35
                    for fidx in sorted_face_indices: pygame.draw.polygon(screen, np.clip(am, 0, 255), screen_pts[Tetrahedron.FACES_NP[fidx]])
                    for i, j in t.EDGES_NP: pygame.draw.line(screen, bc, screen_pts[i], screen_pts[j], 1)
                else:
                    for fidx in sorted_face_indices[::-1]: pygame.draw.polygon(screen, cc[fidx], screen_pts[Tetrahedron.FACES_NP[fidx]])
                    for i, j in t.EDGES_NP: pygame.draw.line(screen, (0,0,0), screen_pts[i], screen_pts[j], 1)

                vert_color = (0,0,0) if (flags['j1'] and not flags['t3']) else (255,255,255)
                for p in screen_pts: pygame.draw.circle(screen, vert_color, p, 1)
                if t.label: surf = font_s.render(t.label, True, (255,255,0)); screen.blit(surf, surf.get_rect(center=cam.project(t.pos + [0, 8, 0])))

        if flags['t3'] and np.dot(norm(cam.pan - (cam.pan - cam_forward)), [0,0,1]) > 0.7:
            pygame.draw.circle(screen, (0,0,0), (WIDTH//2, HEIGHT//2), int((WIDTH / 2.2) * tl))

        now_ticks = pygame.time.get_ticks(); text_color = (0,0,0) if (flags['j1'] and not flags['t3']) else (200,200,200)
        msgs = [m for m in msgs if now_ticks < m[2]]
        for ts, yo, et in msgs:
            s = font_l.render(ts, True, text_color); s.set_alpha(max(0, min(255, (et - now_ticks) * 255 / 1000)))
            screen.blit(s, s.get_rect(center=(WIDTH//2, HEIGHT//2 + yo)))

        for i, (txt, et) in enumerate(list(net_messages)):
            if time.time() > et: net_messages.popleft()
            else:
                s = font_s.render(txt, True, (255, 200, 100)); s.set_alpha(max(0, min(255, (et - time.time()) * 100))); screen.blit(s, (10, 40 + i * 25))

        screen.blit(font_s.render(f"FPS: {int(fps)}", True, (255, 255, 0)), (10, 10))
        zf = DEFAULT_CAM_DIST / cam.dist
        zoom_text = f"{zf:.1f}x" if zf > 10 else f"{zf:.4f}x" if zf < 0.1 else f"{zf:.2f}x"
        mode_text = f"Mode: {game_mode.replace('_',' ').title()} | Tets: {len(world.tets)} | Zoom: {zoom_text} | Z/C Time ({time_scale:.1f}x)"
        if governor_active and time_scale > 1: mode_text += " (GOV)"
        top_leg = font_s.render(mode_text, True, (0,255,255)); screen.blit(top_leg, top_leg.get_rect(center=(WIDTH//2, 20)))
        leg_text = "WASD/RMB View | QE Pan | R/F Zoom | V Save | H Host | TAB Join | ~ Reset | T/X Utils"
        bot_leg = font_s.render(leg_text, True, (0, 255, 255)); screen.blit(bot_leg, bot_leg.get_rect(center=(WIDTH//2, HEIGHT-25)))

        pygame.display.flip()

    if host_instance: host_instance.stop()
    if guest_instance: guest_instance.stop()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
