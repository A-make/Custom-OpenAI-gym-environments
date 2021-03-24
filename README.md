# Custom OpenAI Gym environments

Some custom environments for OpenAI gym.

## Custom environments

- `cartpole_swingup.py` is a modification of the original `cartpole.py`. Now the task is to swing-up the pole and optionally also balance after it swinging up.
- `double_pendulum.py`is a modification of the original `acrobot.py`. Now actuation is applied to the fixed end joint rather than the center joint as it is in acrobot. The environment supports multiple tasks such as swing-up, balance, and swing-up + balance.

Registered environments:
- `CartPoleSwingUp-v0` has been created which is a modification of the original [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/). This custom version includes having to swing the pole to the inverted state before balancing.

For more information on creating custom environments, see [How to create new environments for Gym](docs/creating-environments.md).
## Wrappers

- `acrobot_wrapper.py`: wraps the original `acrobot` environment to support new tasks such as balancing and swing-up + balance. acrobot alone only supports the swing-up task.

Example usage of wrapper:
```python
env = gym.make('Acrobot-v1')
env = AcrobotWrapper(env, task="balance", max_steps=500)
```
## Install
Install package:
```bash
pip install -e gym_custom
```
The `-e` option is for 'editable mode' which allows the code to be updated after the package has been installed.

### Requirements
Tested and working with gym 0.18.0

## Usage
By installing this package, the custom environments get registered (see `gym_custom/envs/__init__.py`) in gym and can be used by:

```Python
import gym
env = gym.make('gym_custom:CartPoleSwingUp-v0')
```