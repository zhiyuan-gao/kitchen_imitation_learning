# Cup-to-Sink ACT Policy

这个仓库包含 `cup_to_sink` 任务的训练脚本，以及已经训练好的 ACT checkpoint。任务是 Franka Panda 把杯子放进水槽。

数据集不放在这个 GitHub 仓库里，需要单独从 Hugging Face 下载：

```bash
git lfs install
git clone https://huggingface.co/datasets/yxzhan/cup_to_sink cup_to_sink
```

## 仓库内容

```text
scripts/train_cup_to_sink_act.sh
scripts/train_cup_to_sink_diffusion.sh
scripts/run_act_inference_example.py
checkpoints/act_100000/pretrained_model/
```

训练好的 ACT checkpoint 在：

```text
checkpoints/act_100000/pretrained_model
```

这个目录需要整体使用，不要只拷贝 `model.safetensors`。推理时还需要同目录下的 `config.json`、`policy_preprocessor.json`、`policy_postprocessor.json` 和两个 normalization 的 `.safetensors` 文件。

## 环境

模型使用 LeRobot `0.4.4` 训练。

```bash
conda create -n lerobot_train python=3.11 -y
conda activate lerobot_train
pip install -r requirements.txt
```

如果手动安装 PyTorch，需要选择和本机 CUDA driver 匹配的版本。

## 数据格式

训练数据是 LeRobot v3.0 格式：

- 100 条 demo
- 34,431 帧
- 25 Hz
- 机器人：`franka_panda`
- 任务文本：`put the cup into the sink`
- 相机：`front`、`left`、`right`、`wrist`
- 图像大小：`256x256`

observation state 和 action 都是 8 维，顺序相同：

```text
panda_joint1, panda_joint2, panda_joint3, panda_joint4,
panda_joint5, panda_joint6, panda_joint7, gripper_width
```

注意：这个模型输出的是 **7 个关节目标位置 + gripper width**，不是 end-effector pose，也不是 delta action。

## 用训练好的 checkpoint 推理

仿真推理只需要 checkpoint 和 LeRobot 环境，不需要下载训练数据集。加载方式如下：

```python
from pathlib import Path
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import predict_action

checkpoint = Path("./checkpoints/act_100000/pretrained_model")
device = torch.device("cuda")

policy = ACTPolicy.from_pretrained(checkpoint)
policy.to(device)
policy.eval()

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=str(checkpoint),
)
```

每个 episode reset 后需要清空 ACT 的内部 action chunk 缓存：

```python
policy.reset()
```

每个仿真 step 准备一个 observation：

```python
obs = {
    "observation.state": state_np,          # shape: (8,)
    "observation.images.front": front_rgb,  # shape: (H, W, 3), uint8, RGB
    "observation.images.left": left_rgb,
    "observation.images.right": right_rgb,
    "observation.images.wrist": wrist_rgb,
}
```

其中 `state_np` 的顺序必须是：

```text
panda_joint1, panda_joint2, panda_joint3, panda_joint4,
panda_joint5, panda_joint6, panda_joint7, gripper_width
```

然后调用：

```python
action = predict_action(
    obs,
    policy,
    device,
    preprocessor,
    postprocessor,
    use_amp=False,
    task="put the cup into the sink",
    robot_type="franka_panda",
)
```

`action` 是一个 shape 为 `(1, 8)` 的 tensor。把 batch 维去掉后，按下面顺序发给 Franka Panda 的 joint position controller：

```text
panda_joint1, panda_joint2, panda_joint3, panda_joint4,
panda_joint5, panda_joint6, panda_joint7, gripper_width
```

建议仿真控制频率接近训练数据的 `25 Hz`。如果相机输出不是 `256x256`，最好在仿真侧 resize 到 `256x256`，并保持 RGB 通道顺序。

## 最小仿真循环示例

下面是接入仿真的伪代码：

```python
policy.reset()
obs = sim.reset()

for step in range(max_steps):
    lerobot_obs = {
        "observation.state": obs["state"],
        "observation.images.front": obs["front_rgb"],
        "observation.images.left": obs["left_rgb"],
        "observation.images.right": obs["right_rgb"],
        "observation.images.wrist": obs["wrist_rgb"],
    }

    action = predict_action(
        lerobot_obs,
        policy,
        device,
        preprocessor,
        postprocessor,
        use_amp=False,
        task="put the cup into the sink",
        robot_type="franka_panda",
    )

    action_np = action.squeeze(0).cpu().numpy()
    obs, reward, done, info = sim.step(action_np)

    if done:
        break
```

eval 成功率需要仿真环境自己提供 reset 和 success 判断，比如杯子是否进入 sink 区域、是否稳定、机器人是否没有异常碰撞或关节超限。

## 离线推理检查

如果想先确认 checkpoint 能正常加载，并在训练集某一帧上跑一次推理，可以下载数据集后执行：

```bash
python scripts/run_act_inference_example.py \
  --dataset-root ./cup_to_sink \
  --checkpoint ./checkpoints/act_100000/pretrained_model \
  --sample-index 0
```

这个脚本会打印模型预测的 8 维 action，以及同一帧数据集里的 demo action。这个脚本只是检查加载和输入输出格式，不等价于仿真 rollout 成功率。

## 重新训练

训练 ACT：

```bash
bash scripts/train_cup_to_sink_act.sh
```

训练 Diffusion Policy：

```bash
bash scripts/train_cup_to_sink_diffusion.sh
```

脚本默认使用：

```text
ROOT_DIR=<repo root>
DATASET_ROOT=${ROOT_DIR}/cup_to_sink
OUTPUT_DIR=${ROOT_DIR}/outputs/train/...
LEROBOT_TRAIN=lerobot-train
DEVICE=cuda
```

可以用环境变量覆盖：

```bash
DATASET_ROOT=/path/to/cup_to_sink \
OUTPUT_DIR=/path/to/output \
LEROBOT_TRAIN=/path/to/lerobot-train \
DEVICE=cuda \
bash scripts/train_cup_to_sink_act.sh
```

## Checkpoint 信息

当前包含的 ACT checkpoint：

```text
checkpoints/act_100000/pretrained_model
```

训练步数：`100000`

最终训练 loss：约 `0.046`

checkpoint 目录大小：约 `60 MB`
