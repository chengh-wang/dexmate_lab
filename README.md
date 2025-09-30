# Dexmate_lab


## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).


- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    python -m pip install -e source/dexmate_lab
- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        ```bash
        python scripts/list_envs.py
        ```

    - Running a training:

        ```bash
        python scripts/rsl_rl/train.py --task Template-Dexmate-Lab-v0 --num_envs 512 --headless
        ```

    - Evaluate a trained policy
        ```bash
        python scripts/rsl_rl/play.py --task Template-Dexmate-Lab-v0 --num_envs 1 --load_run 2025-09-29_00-21-52
        ```
## Task Overview

This environment train a cube lifting task using the Vega robot's right arm and hand to grasp and lift a 6cm cube (100g) to a target position.

### Actions

**Action Space:** 12-dimensional continuous control
- **Right arm joints** (7 DoF): `R_arm_j1` through `R_arm_j7`
- **Hand joints** (5 DoF): `R_th_j0`, `R_th_j1`, `R_ff_j1`, `R_mf_j1`, `R_rf_j1`, `R_lf_j1`


### Observations
**1. Policy Observations**
- Object orientation (quaternion in robot base frame)
- Target object pose command
- Last action

**2. Proprioception Observations**
- Joint positions and velocities (12 joints)
- Hand tip states (position, velocity, orientation of wrist + 5 fingertips)
- Contact forces on all 5 fingertips (clipped to ±20N)

**3. Perception Observations**
- Object point cloud (64 points in robot base frame, clipped to ±2m)

### Rewards

The reward encourages successful grasping (through Reaching and Grasping) and lifting (through Lifting and Success), by changing “binary_contact” flag from grasping function, we can selecting between two finger grasp or full hand grasp:

| Component | Description |
|-----------|-------------|
| **Reaching** | Exponential reward for minimizing finger-to-object distance |
| **Grasping** |  Binary reward for proper grasp (thumb + any finger contact > 0.5N) |
| **Lifting** | osition tracking to target, gated by contact presence |
| **Success** |  High reward when object reaches target position (within 0.1m) |
| **Action penalty** | L2 penalty on actions |
| **Action rate penalty** |  L2 penalty on action changes |
| **Early termination** |  Penalty for dropping the object |

### Curriculum Learning

- **Gravity curriculum:** Progressively introduces gravity as training advances
  - Initial: No gravity (0 m/s²), the cube floating around the env.
  - Final: Standard Earth gravity (-9.81 m/s²).

### Termination Conditions

- **Timeout:** 5 seconds 
- **Out of bounds:** Object exits the workspace


### Environment Randomization

- **Table height:** Randomized between 0.05-0.15m per episode
- **Object spawn:** Random position on table surface (x: 0-0.25m, y: -0.2 to 0.15m)
- **Target pose:** Random reachable position (x: 0.2-0.3m, y: -0.1 to 0.1m, z: 0.4-0.5m)

