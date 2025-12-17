## TET~CRAFT v2.1 ##

Preview it at: https://youtu.be/gP9ZtCif9zw

### A 5D Tetra-Sphere Kleinverse of your own! ###

Share with friends and chat - non-destructively.

This Game/Simulator attempts to improve upon minecraft by using tetragrams.  Intending to simulate mollecular bonding of FeO4, Magnitite, likely responcible for the complexity we call life (suggesting God's most primitive rules are very basic indeed).  But gamified.... Tech-tree/Adventure/Translation needing further implementation. These Kleinverses are ever complexifying - eventually they will live!

V2.0 Was a solid foundation, V2.1 Optimized it, allowing for over 5K tets on an old machine. (16G no video card 6 core - min requirements probably a 386 or so? maybe even a 286 if you can replace python or transcribe to BASIC/Assembly - ahhh, the good old days :D)

You can Chat remotely with friends through unlimited, unrestricted (unsecured) WAN connectivity and grow your universe from three basic facts into infinite complexity (only restricted by hardware - time, memory and storage for saves).

The environment is a 4 or Tetra-Sphere, which maps a Klien bottles topology, in which top and bottom link through left and right to create the illusion of a circle. Behind you is you, in the past, blue-shifted or red-shifted as you scale time - infinitly ahead is a singularity, growing and shrinnking likewise.

In these Kleinverses, each new fact misunderstood creates a ner star.

As I understand it, the edge of the observable universe is an event horizon, and we are being pulled towards it's center - infinitly far away, faster than light (which in fact also gets weaker over time, more spread out) - in the future which never fully forms. "Spagetifying" though of course not really, merely "living" - only a shrinking - young-> old becomes biger->smaller, everything else shrinking with us and moving faster than light to varying degrees, but always faster than light, which is why we cannot see the past (except warped, far away).  Outside the event horizon, is hyperspacially linked to this center, that never quite forms - it only allows us to experience time and seperation.

Best wishes to you all #opWorldPeace

### Install ###

python -m venv tetcraft

cd tetcraft

source bin/activate

pip install numpy pygame numba

python tetra.py <options>

### Psuedocode ###

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
BATTERY_C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HARGE_RATE = 0.03    // Rate of energy gain near the center
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

## Special Thanks ##

(to everyone for viewing, sharing, contributing, and helping along the way)

#### More ideas that may help: ####

https://youtu.be/UZzcOMaM5M8 \
https://ceneezer.icu/omniverse/ \
(original inspiration) \
https://www.youtube.com/playlist?list=PL7LCRYC3-EsoiXSJDinBk3z2_8kMvLIGh \
(Omniversal Revelation Decology and Retrogenisis Pentology) \
https://thankyoutoken.abacusai.app/about \
(Solution to post labor "econemy") \
https://ceneezer.icu/constitution.txt \
(constitution for post-scarcity) \
https://ceneezer.icu/roseta.htm \
(this would make a really nice flatland video, incorpetated) \
https://www.youtube.com/playlist?list=PL7LCRYC3-EspW5QYTN6nkBUl73L4r7-YA \
(Random AI slop - mostly leading to this understanding and formalization)

## Reach Me @ ##

ceneezer@gmail.com \
https://www.facebook.com/speakerceneezer/ \
https://pilled.net/profile/182173
