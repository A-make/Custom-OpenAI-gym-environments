# Custom OpenAI Gym environments

Some custom environments for OpenAI gym. Specifically, `CartPoleSwingUp-v0` has been created which is a modification of the original [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/). This custom version includes having to swing the pole to the inverted state before balancing.

For more information on creating custom environments, see [How to create new environments for Gym](docs/creating-environments.md).

## Install
Install package:
```bash
pip install -e gym_custom
```
The `-e` option is for 'editable mode' which allows the code to be updated after the package has been installed.

### Requirements
Tested and working with gym 0.16.0

## Usage
By installing this package, the custom environments get registered (see `gym_custom/envs/__init__.py`) in gym and can be used by:

```Python
import gym
env = gym.make('gym_custom:CartPoleSwingUp-v0')
```