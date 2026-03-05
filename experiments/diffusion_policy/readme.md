# Reproducing Diffusion Policy

## 1. Training Hyperparameters

### 1.1 `epoch_every_n_steps`

- **Definition**: The number of gradient steps defined as a single "epoch".
- **Notes**: 
  - Since datasets are stored as complete trajectories, a single pass through the data can be extensive. Defining a custom step-based epoch allows for more frequent logging.
  - Use **Rollout** evaluation to monitor training progress at a higher frequency than the defined epoch interval.
  - The Cosine Learning Rate Scheduler is typically synchronized with these defined epochs.

### 1.2 `frame_stack`

- **Value**: 2
- **Definition**: The number of historical observations concatenated to form the input state (also referred to as the **Observation Horizon**).

---

## 2. Algorithm & Architecture

### 2.1 Learning Rate & Scheduler

- **Warmup Steps**: 500 (linear ramp-up to the initial learning rate).
- **Num Cycles**: 0.5 (cosine decay from the peak learning rate to zero).

### 2.2 Horizon Settings

- **Observation Horizon**: 2 
- **Action Horizon**: 8 
- **Prediction Horizon**: 16 

### 2.3 U-Net Structure

- **Diffusion Conditioning**:
  - **Diffusion Step Embedding**: A sinusoidal embedding representing the current noise level, injected to guide the model across different denoising timesteps.
  - **Global Features**: Visual encoder outputs concatenated with proprioceptive condition features.
  - **FiLM Integration**: Global features are projected and split into scale and bias parameters to modulate the residual blocks within the U-Net.
- **Hyperparameters**:
  - `down_dims`: [256, 512, 1024]  (Hidden dimensions for action feature processing).
  - `kernel_size`: 5  (1D convolutional kernel size for temporal processing).
  - **Normalization Choice (**`n_groups=8`**)**:  GroupNorm is preferred over BatchNorm/LayerNorm for the following reasons:
    - *Small Batch Size*: BatchNorm performance degrades significantly when the batch size is small, which is common in robotic trajectory training.
    - *Time-Conditioning*: BatchNorm normalize across different timesteps, introducing undesired correlations.
    - *Channel Independence*: Unlike LayerNorm, which aggregates across all channels (potentially conflating distinct joint dimensions), GroupNorm preserves specific feature characteristics by normalizing within groups.

---

## 3. Dataset Structure

### 3.1 Observation Keys

Run `python get_dataset_info.py --dataset <path_to_hdf5>` to inspect the observation space.

- **Obs Structure**: Contains proprioceptive data such as `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, `object`, etc.

```bash
key: obs
    observation key object with shape (59, 10)
    observation key robot0_eef_pos with shape (59, 3)
    observation key robot0_eef_quat with shape (59, 4)
    observation key robot0_eef_quat_site with shape (59, 4)
    observation key robot0_gripper_qpos with shape (59, 2)
    observation key robot0_gripper_qvel with shape (59, 2)
    observation key robot0_joint_pos with shape (59, 7)
    observation key robot0_joint_pos_cos with shape (59, 7)
    observation key robot0_joint_pos_sin with shape (59, 7)
    observation key robot0_joint_vel with shape (59, 7)
```

### 3.2 Configuring Observation Keys

- To configure your model, select the specific keys required for your task. The following four keys are typically sufficient for a single-arm setup:

```json
{
    "observation": {
        "modalities": {
            "obs": {
                "low_dim": [
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                    "object"
                ],
                "rgb": [],
                "depth": [],
                "scan": []
            }
        }
    }
}
```

- Shown in the terminal output:

```bash
============= Initialized Observation Utils with Obs Spec =============

using obs modality: low_dim with keys: ['robot0_eef_quat', 'object', 'robot0_gripper_qpos', 'robot0_eef_pos']
using obs modality: rgb with keys: []
using obs modality: depth with keys: []
using obs modality: scan with keys: []

============= Loaded Environment Metadata =============
obs key object with shape (10,)
obs key robot0_eef_pos with shape (3,)
obs key robot0_eef_quat with shape (4,)
obs key robot0_gripper_qpos with shape (2,)
```

