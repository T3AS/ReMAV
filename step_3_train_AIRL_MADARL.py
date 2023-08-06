#!/bin/env python
import gym
import macad_gym  # noqa F401
import argparse
import os
from pprint import pprint
from network_models import d_net, Policy_net
from algo import Discriminator, PPO, AIRL_wrapper

import cv2
import ray
import ray.tune as tune
from gym.spaces import Box, Discrete
from macad_agents.rllib.env_wrappers import wrap_deepmind
from macad_agents.rllib.models import register_mnih15_net
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents import a3c
from ray.rllib.agents import dqn
import numpy as np
# from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy
import datetime
import json
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy #0.8.5
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.tune import register_env
import time
from pprint import pprint
import pandas as pd

from tqdm import tqdm
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import gc
tf.disable_v2_behavior()


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
model = VGG16(weights='imagenet', include_top=False)

from tensorboardX import SummaryWriter
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
writer = SummaryWriter("logss/" + timestamp)



# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# tf.keras.backend.set_session(tf.Session(config=config));

try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class



parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    default="PongNoFrameskip-v4",
    help="Name Gym env. Used only in debug mode. Default=PongNoFrameskip-v4")
parser.add_argument(
    "--checkpoint-path",
    #Replace it with your path of last training checkpoints
    default='/home/aizaz/ray_results/Training_3/MA-Inde-PPO-SSUI3CCARLA/PPO_HomoNcomIndePOIntrxMASS3CTWN3-v0_0_2021-06-03_21-26-554514dd5k/checkpoint_300/checkpoint-300',
    help="Path to checkpoint to resume training")
parser.add_argument(
    "--disable-comet",
    action="store_true",
    help="Disables comet logging. Used for local smoke tests")
parser.add_argument(
    "--num-workers",
    default=1, #2
    type=int,
    help="Num workers (CPU cores) to use")
parser.add_argument(
    "--num-gpus", default=1, type=int, help="Number of gpus to use. Default=2")
parser.add_argument(
    "--sample-bs-per-worker",
    default=1024,
    type=int,
    help="Number of samples in a batch per worker. Default=50")
parser.add_argument(
    "--train-bs",
    default=128,
    type=int,
    help="Train batch size. Use as per available GPU mem. Default=500")
parser.add_argument(
    "--envs-per-worker",
    default=1,
    type=int,
    help="Number of env instances per worker. Default=10")
parser.add_argument(
    "--notes",
    default=None,
    help="Custom experiment description to be added to comet logs")
parser.add_argument(
    "--model-arch",
    default="mnih15",
    help="Model architecture to use. Default=mnih15")
parser.add_argument(
    "--num-steps",
    default=2000000, 
    type=int,
    help="Number of steps to train. Default=20M")
parser.add_argument(
    "--num-iters",
    default=1, #20
    type=int,
    help="Number of training iterations. Default=20")
parser.add_argument(
    "--log-graph",
    action="store_true",
    help="Write TF graph on Tensorboard for debugging")
parser.add_argument(
    "--num-framestack",
    type=int,
    default=4,
    help="Number of obs frames to stack")
parser.add_argument(
    "--redis-address",
    default=None,
    help="Address of ray head node. Be sure to start ray with"
    "ray start --redis-address <...> --num-gpus<.> before running this script")
parser.add_argument(
    "--use-lstm", action="store_true", help="Append a LSTM cell to the model")

parser.add_argument('--logdir', help='log directory', default='log/')
parser.add_argument('--savedir', help='save directory', default='trained_models/')
parser.add_argument('--gamma', default=0.95, type=float)
parser.add_argument('--iters', default=int(1e4), type=int)

args = parser.parse_args()

#--------------------------------------------------------------------
model_name = args.model_arch
if model_name == "mnih15":
    register_mnih15_net()  # Registers mnih15
else:
    print("Unsupported model arch. Using default")
    register_mnih15_net()
    model_name = "mnih15"

# Used only in debug mode
env_name = "HomoNcomIndePOIntrxMASS3CTWN3-v0"
env = gym.make(env_name)
env_actor_configs = env.configs
num_framestack = args.num_framestack
# env_config["env"]["render"] = False
#--------------------------------------------------------------------

def env_creator(env_config):
    # NOTES: env_config.worker_index & vector_index are useful for
    # curriculum learning or joint training experiments
    import macad_gym
    env = gym.make("HomoNcomIndePOIntrxMASS3CTWN3-v0")

    # Apply wrappers to: convert to Grayscale, resize to 84 x 84,
    # stack frames & some more op
    env = wrap_deepmind(env, dim=84, num_framestack=num_framestack)
    return env


register_env(env_name, lambda config: env_creator(config))
#--------------------------------------------------------------------

# Placeholder to enable use of a custom pre-processor
class ImagePreproc(Preprocessor):
    def _init_shape(self, obs_space, options):
        self.shape = (84, 84, 3)  # Adjust third dim if stacking frames
        return self.shape

    def transform(self, observation):
        observation = cv2.resize(observation, (self.shape[0], self.shape[1]))
        return observation
def transform(self, observation):
        observation = cv2.resize(observation, (self.shape[0], self.shape[1]))
        return observation

ModelCatalog.register_custom_preprocessor("sq_im_84", ImagePreproc)
#--------------------------------------------------------------------

if args.redis_address is not None:
    # num_gpus (& num_cpus) must not be provided when connecting to an
    # existing cluster
    ray.init(redis_address=args.redis_address,object_store_memory=10**9)
else:
    ray.init(num_gpus=args.num_gpus,object_store_memory=10**9)

config = {
    # Model and preprocessor options.
    "model": {
        "custom_model": model_name,
        "custom_options": {
            # Custom notes for the experiment
            "notes": {
                "args": vars(args)
            },
        },
        # NOTE:Wrappers are applied by RLlib if custom_preproc is NOT specified
        "custom_preprocessor": "sq_im_84",
        "dim": 84,
        "free_log_std": False,  # if args.discrete_actions else True,
        "grayscale": True,
        # conv_filters to be used with the custom CNN model.
        # "conv_filters": [[16, [4, 4], 2], [32, [3, 3], 2], [16, [3, 3], 2]]
    },
    # preproc_pref is ignored if custom_preproc is specified
    # "preprocessor_pref": "deepmind",

    # env_config to be passed to env_creator
    
    "env_config": env_actor_configs
}

def default_policy():
    env_actor_configs["env"]["render"] = False

    config = {
    # Model and preprocessor options.
    "model": {
        "custom_model": model_name,
        "custom_options": {
            # Custom notes for the experiment
            "notes": {
                "args": vars(args)
            },
        },
        # NOTE:Wrappers are applied by RLlib if custom_preproc is NOT specified
        "custom_preprocessor": "sq_im_84",
        "dim": 84,
        "free_log_std": False,  # if args.discrete_actions else True,
        "grayscale": True,
        # conv_filters to be used with the custom CNN model.
        # "conv_filters": [[16, [4, 4], 2], [32, [3, 3], 2], [16, [3, 3], 2]]
    },


    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # The GAE(lambda) parameter.
    "lambda": 1.0,
    # Initial coefficient for KL divergence.
    "kl_coeff": 0.3,
    # Size of batches collected from each worker.
    "rollout_fragment_length": 128,
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    # "train_batch_size": 4000,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 64,
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": True,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 8,
    # Stepsize of SGD.
    "lr": 5e-5,
    # Learning rate schedule.
    # "lr_schedule": None,
    # Share layers for value function. If you set this to True, it's important
    # to tune vf_loss_coeff.
    "vf_share_layers": False,
    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set vf_share_layers: True.
    "vf_loss_coeff": 1.0,
    # Coefficient of the entropy regularizer.
    "entropy_coeff": 0.1,
    # Decay schedule for the entropy regularizer.
    "entropy_coeff_schedule": None,
    # PPO clip parameter.
    "clip_param": 0.3,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,
    # If specified, clip the global norm of gradients by this amount.
    "grad_clip": None,
    # Target value for KL divergence.
    "kl_target": 0.03,
    # Whether to rollout "complete_episodes" or "truncate_episodes".
    "batch_mode": "complete_episodes",
    # Which observation filter to apply to the observation.
    "observation_filter": "NoFilter",
    # Uses the sync samples optimizer instead of the multi-gpu one. This is
    # usually slower, but you might want to try it if you run into issues with
    # the default optimizer.
    "simple_optimizer": False,
    # Use PyTorch as framework?
    "use_pytorch": False,

    # Discount factor of the MDP.
    "gamma": 0.99,
    # Number of steps after which the episode is forced to terminate. Defaults
    # to `env.spec.max_episode_steps` (if present) for Gym envs.
    "horizon": 512,
    # Calculate rewards but don't reset the environment when the horizon is
    # hit. This allows value estimation and RNN state to span across logical
    # episodes denoted by horizon. This only has an effect if horizon != inf.
    "soft_horizon": True,
    # Don't set 'done' at the end of the episode. Note that you still need to
    # set this if soft_horizon=True, unless your env is actually running
    # forever without returning done=True.
    "no_done_at_end": True,
    "monitor": False,




    # System params.
    # Should be divisible by num_envs_per_worker
    "sample_batch_size":
     args.sample_bs_per_worker,
    "train_batch_size":
    args.train_bs,
    # "rollout_fragment_length": 128,
    "num_workers":
    args.num_workers,
    # Number of environments to evaluate vectorwise per worker.
    "num_envs_per_worker":
    args.envs_per_worker,
    "num_cpus_per_worker":
    1,
    "num_gpus_per_worker":
    1,
    # "eager_tracing": True,

    # # Learning params.
    # "grad_clip":
    # 40.0,
    # "clip_rewards":
    # True,
    # either "adam" or "rmsprop"
    "opt_type":
    "adam",
    # "lr":
    # 0.003,
    "lr_schedule": [
        [0, 0.0006],
        [20000000, 0.000000000001],  # Anneal linearly to 0 from start 2 end
    ],
    # rmsprop considered
    "decay":
    0.5,
    "momentum":
    0.0,
    "epsilon":
    0.1,
    # # balancing the three losses
    # "vf_loss_coeff":
    # 0.5,  # Baseline loss scaling
    # "entropy_coeff":
    # -0.01,

    # preproc_pref is ignored if custom_preproc is specified
    # "preprocessor_pref": "deepmind",
   # "gamma": 0.99,

    "use_lstm": args.use_lstm,
    # env_config to be passed to env_creator
    "env":{
        "render": True
    },
    # "in_evaluation": True,
    # "evaluation_num_episodes": 1,
    "env_config": env_actor_configs
    }


    # pprint (config)
    return (PPOTFPolicy, Box(0.0, 255.0, shape=(84, 84, 3)), Discrete(9),config)

pprint (args.checkpoint_path)
pprint(os.path.isfile(args.checkpoint_path))




#--------------------------------------------------------------------
multiagent = True

trainer = ppo.PPOTrainer(
    env=env_name,
    # Use independent policy graphs for each agent
    config={

        "multiagent": {
            "policies": {
                id: default_policy()
                for id in env_actor_configs["actors"].keys()
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },
        "env_config": env_actor_configs,
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.envs_per_worker,
        "sample_batch_size": args.sample_bs_per_worker,
        "rollout_fragment_length": args.sample_bs_per_worker,

        "train_batch_size": args.train_bs,
 
    })

if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
    trainer.restore(args.checkpoint_path)
    print("Loaded checkpoint from:{}".format(args.checkpoint_path))
# pprint (trainer.config)
#print (dir(trainer))

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

info_car2PPO_scenario_1  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_1.json', lines=True)
info_car2PPO_scenario_2  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_2.json', lines=True)
info_car2PPO_scenario_3  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_3.json', lines=True)
info_car2PPO_scenario_4  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_4.json', lines=True)
info_car2PPO_scenario_5  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_5.json', lines=True)
info_car2PPO_scenario_6  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_6.json', lines=True)
info_car2PPO_scenario_7  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_7.json', lines=True)
info_car2PPO_scenario_8  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_8.json', lines=True)
info_car2PPO_scenario_9  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_9.json', lines=True)
info_car2PPO_scenario_10  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_10.json', lines=True)
info_car2PPO_scenario_11  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_11.json', lines=True)
info_car2PPO_scenario_12  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_12.json', lines=True)
info_car2PPO_scenario_13  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_13.json', lines=True)
info_car2PPO_scenario_14  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_14.json', lines=True)
info_car2PPO_scenario_15  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_15.json', lines=True)
info_car2PPO_scenario_16  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_16.json', lines=True)
info_car2PPO_scenario_17  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_17.json', lines=True)
info_car2PPO_scenario_18  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_18.json', lines=True)
info_car2PPO_scenario_19  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_19.json', lines=True)
info_car2PPO_scenario_20  = pd.read_json ('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/three_way_episode_20.json', lines=True)



obs_dims = (12800,)
n_actions = 9


agent = PPO(args.savedir, Policy_net, obs_dims, n_actions)
D = Discriminator(args.savedir, obs_dims, n_actions)
# env = gym.make("CartPole-v0")
# trainerAIRL = AIRL_wrapper(agent, D, env, args.savedir)
# trainerAIRL.train(args.iters)


sess = tf.Session() 
sess.run(tf.global_variables_initializer())
d_save_dir = os.path.join(args.savedir, "disc_three_way")
g_save_dir = os.path.join(args.savedir, "gen")
os.makedirs(d_save_dir, exist_ok=True)
os.makedirs(g_save_dir, exist_ok=True)



path_checkpoint = "/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/trained_models/disc_three_way"

# with tf.Session() as sess:
#   saver = tf.train.import_meta_graph('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/trained_models/disc/20.meta')
#   saver.restore(sess, tf.train.latest_checkpoint('/home/aizaz/Desktop/PhD-20210325T090933Z-001/PhD/1_March_2023/Journal_Paper/examples/trained_models/disc'))

# """Load model from `save_path` if there exists."""
# latest_checkpoint = tf.train.latest_checkpoint(path_checkpoint)

# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     saver.restore(sess, tf.train.latest_checkpoint('./'))





12800



train_fq = 32
rewards_his = []


expert_observations_1 = info_car2PPO_scenario_1['state'].to_list()
expert_actions_1 = info_car2PPO_scenario_1['action'].to_list()
expert_observations_2 = info_car2PPO_scenario_2['state'].to_list()
expert_actions_2 = info_car2PPO_scenario_2['action'].to_list()
expert_observations_3 = info_car2PPO_scenario_3['state'].to_list()
expert_actions_3 = info_car2PPO_scenario_3['action'].to_list()
expert_observations_4 = info_car2PPO_scenario_4['state'].to_list()
expert_actions_4 = info_car2PPO_scenario_4['action'].to_list()
expert_observations_5 = info_car2PPO_scenario_5['state'].to_list()
expert_actions_5 = info_car2PPO_scenario_5['action'].to_list()
expert_observations_6 = info_car2PPO_scenario_6['state'].to_list()
expert_actions_6 = info_car2PPO_scenario_6['action'].to_list()
expert_observations_7 = info_car2PPO_scenario_7['state'].to_list()
expert_actions_7 = info_car2PPO_scenario_7['action'].to_list()
expert_observations_8 = info_car2PPO_scenario_8['state'].to_list()
expert_actions_8 = info_car2PPO_scenario_8['action'].to_list()
expert_observations_9 = info_car2PPO_scenario_9['state'].to_list()
expert_actions_9 = info_car2PPO_scenario_9['action'].to_list()
expert_observations_10 = info_car2PPO_scenario_10['state'].to_list()
expert_actions_10 = info_car2PPO_scenario_10['action'].to_list()
expert_observations_11 = info_car2PPO_scenario_11['state'].to_list()
expert_actions_11 = info_car2PPO_scenario_11['action'].to_list()
expert_observations_12 = info_car2PPO_scenario_12['state'].to_list()
expert_actions_12 = info_car2PPO_scenario_12['action'].to_list()
expert_observations_13 = info_car2PPO_scenario_13['state'].to_list()
expert_actions_13 = info_car2PPO_scenario_13['action'].to_list()
expert_observations_14 = info_car2PPO_scenario_14['state'].to_list()
expert_actions_14 = info_car2PPO_scenario_14['action'].to_list()
expert_observations_15 = info_car2PPO_scenario_15['state'].to_list()
expert_actions_15 = info_car2PPO_scenario_15['action'].to_list()
expert_observations_16 = info_car2PPO_scenario_16['state'].to_list()
expert_actions_16 = info_car2PPO_scenario_16['action'].to_list()
expert_observations_17 = info_car2PPO_scenario_17['state'].to_list()
expert_actions_17 = info_car2PPO_scenario_17['action'].to_list()
expert_observations_18 = info_car2PPO_scenario_18['state'].to_list()
expert_actions_18 = info_car2PPO_scenario_18['action'].to_list()
expert_observations_19 = info_car2PPO_scenario_19['state'].to_list()
expert_actions_19 = info_car2PPO_scenario_19['action'].to_list()
expert_observations_20 = info_car2PPO_scenario_20['state'].to_list()
expert_actions_20 = info_car2PPO_scenario_20['action'].to_list()

obs_his = []
act_his = []
r_his = []
v_preds_next_his = []
v_preds_his = []
dloss_his = []
aloss_his = []
d_reward_his = []

agents_reward_dict = {}

# obs = env.reset()
#for ep in range(2):
step = 0


episode_reward = 0
info_dict= []
img_data = 0
done = False
i=0

py_measurements = {
            "state":0,
            "action": 0,

#            "next_state":0,
        }

action = {}

observations = []
actions = []
rewards = []
v_preds = []
r = 0
obs = env.reset() 
expert_observations = expert_observations_1
expert_actions = expert_actions_1

for j in range(100): #(20*15 episodes):
    i=0
    
    print ("Starting a single episode for testing")



    if (j==5):
        print ("2nd Scenario")
        expert_observations = expert_observations_2
        expert_actions = expert_actions_2
    elif (j==10):
        expert_observations = expert_observations_3
        expert_actions = expert_actions_3
    elif (j==15):
        expert_observations = expert_observations_4
        expert_actions = expert_actions_4
    elif (j==20):
        expert_observations = expert_observations_5
        expert_actions = expert_actions_5
    elif (j==25):
        expert_observations = expert_observations_6
        expert_actions = expert_actions_6
    elif (j==30):
        expert_observations = expert_observations_7
        expert_actions = expert_actions_7
    elif (j==35):
        expert_observations = expert_observations_8
        expert_actions = expert_actions_8
    elif (j==40):
        expert_observations = expert_observations_9
        expert_actions = expert_actions_9
    elif (j==45):
        expert_observations = expert_observations_10
        expert_actions = expert_actions_10
    elif (j==50):
        expert_observations = expert_observations_11
        expert_actions = expert_actions_11
    if (j==55):
        expert_observations = expert_observations_12
        expert_actions = expert_actions_12
    elif (j==60):
        expert_observations = expert_observations_13
        expert_actions = expert_actions_13
    elif (j==65):
        expert_observations = expert_observations_14
        expert_actions = expert_actions_14
    elif (j==70):
        expert_observations = expert_observations_15
        expert_actions = expert_actions_15
    elif (j==75):
        expert_observations = expert_observations_16
        expert_actions = expert_actions_16
    elif (j==80):
        expert_observations = expert_observations_17
        expert_actions = expert_actions_17
    elif (j==85):
        expert_observations = expert_observations_18
        expert_actions = expert_actions_18
    elif (j==90):
        expert_observations = expert_observations_19
        expert_actions = expert_actions_19
    elif (j==95):
        expert_observations = expert_observations_20
        expert_actions = expert_actions_20
    



    while i < 800:  # number of steps in a episodic run 
        i += 1


        
  

        for agent_id, agent_obs in obs.items():
            policy_id = trainer.config["multiagent"]["policy_mapping_fn"](agent_id)

            if (agent_id=="car3"):
                img_data = np.expand_dims(agent_obs, axis=0)
                vgg16_feature = model.predict(img_data)
                
                act, v_pred = agent.act(vgg16_feature.flatten().reshape(-1,12800))
                act = act[0]
            
            action[agent_id] = trainer.compute_action(agent_obs, policy_id=policy_id)
            
                
    
        obs, reward, done, info = env.step({'car1':  action['car1'], 'car2': action['car2'], 'car3': act})

        img_data = np.expand_dims(obs["car3"], axis=0)
        vgg16_feature = model.predict(img_data)
        # observations.append(obs["car2"].flatten()) #reshape(-1,84672))
        
        observations.append(vgg16_feature.flatten()) #reshape(-1,84672))
        actions.append(act)
        rewards.append(reward["car3"])
        v_preds.append(v_pred)
        #obs = next_obs["car2"]
        r += reward["car3"]

        #pprint (next_obs)
        # next_obs = np.stack([obs["car2"]]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
        next_obs = np.stack([vgg16_feature.flatten()]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs

        _, v_pred = agent.act(obs=next_obs.reshape(-1,12800), stochastic=True)
        v_preds_next = v_preds[1:] + [np.asscalar(v_pred)]
        rewards_his.append(r)
        obs_his.extend(observations)
        act_his.extend(actions)
        r_his.extend(rewards)
        v_preds_next_his.extend(v_preds_next)
        v_preds_his.extend(v_preds)


        #print ("Length of expert_observations : ",len(expert_observations[0]))
        #print ("Length of obs_his : ",len(obs_his[0]))
        #print ("Length of expert_actions : ",len(expert_actions))
        #print ("Length of act_his : ",len(act_his))

        # update
        # train discriminator
        if (i+1) % train_fq == 0:
            for _ in range(2):
                _, dloss = D.train(expert_s=expert_observations,
                expert_a=expert_actions,
                agent_s=obs_his,
                agent_a=act_his)
                dloss_his.append(np.mean(dloss))
                d_rewards = D.get_rewards(agent_s=obs_his, agent_a=act_his).reshape(-1)
                d_reward_his.append(np.mean(d_rewards))
                gaes = agent.get_gaes(rewards=d_rewards, v_preds=v_preds_his,v_preds_next=v_preds_next_his)

                gaes = np.array(gaes).reshape(-1)
                v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
        
            # train policy
            inp = [obs_his, act_his, gaes, d_rewards, v_preds_next_his]
            agent.assign_policy_parameters()

            for _ in range(16): #6
                sample_indices = np.random.randint(low=0, high=len(observations), size=128)  
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                _ = agent.train(obs=sampled_inp[0],
                                            actions=sampled_inp[1],
                                            gaes=sampled_inp[2],
                                            rewards=sampled_inp[3],
                                            v_preds_next=sampled_inp[4])

        
            obs_his = []
            act_his = []
            r_his = []
            v_preds_next_his = []
            v_preds_his = []
            # print ("Done")

        #if (i+1) % 10 == 0:
        #    r_eval = eval(10)
        #    print("eval r : {}".format(r_eval))
                    


        # print (i, " ------ ")
    observations = []
    actions = []
    rewards = []
    v_preds = []
    r = 0
    

    gc.collect()
    env._clear_server_state()
    obs = env.reset()  
    print (j, " ------ ")



# saver = tf.train.Saver(save_relative_paths=True)

# save_path = D.saver.save(sess, d_save_dir+"/model.ckpt",write_meta_graph=True)
agent.saver.save(agent.sess, os.path.join(g_save_dir, "{}".format(j)))
D.saver.save(D.sess, os.path.join(d_save_dir, "{}".format(j)),write_meta_graph=True)
print("Model saved")


# done = done['__all__']
env.close()
writer.close()
ray.shutdown()