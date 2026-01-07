import numpy as np
import torch
import os
import random
import h5py
import sys
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations, batch_rotations, transformations, trajectories
from torchvision import transforms, utils
import albumentations as A
from albumentations.pytorch import ToTensorV2
import seaborn as sns
from tqdm import tqdm
import json
import time
path_to_suturebot = os.getenv("PATH_TO_SUTUREBOT")

if path_to_suturebot:
    sys.path.append(os.path.join(path_to_suturebot, 'src'))
from aloha_pro.aloha_scripts.utils import initialize_model_and_tokenizer, encode_text
from img_aug import DataAug
import IPython
e = IPython.embed

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def shift_image(image, shift_x, shift_y):
    """Shift the image by the given x and y offsets."""
    (h, w) = image.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, M, (w, h))
    return shifted_image

def rotate_image(image, angle):
    """Rotate the image by the given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated



class EpisodicDatasetDvrkGeneric(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        tissue_sample_ids, 
        dataset_dir, 
        camera_names, 
        camera_file_suffixes, 
        # num_episodes,
        task_config,
        chunk_size=100,
        norm_stats=None,
        max_len=None,
        command_list=None,
        use_language=False,
        language_encoder="distilbert",
        ):

        super(EpisodicDatasetDvrkGeneric).__init__()

        if len(tissue_sample_ids) == 0:
            raise ValueError("No tissue samples found in the dataset directory.")
        
        # self.episode_ids = episode_ids
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0]
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.camera_file_suffixes = camera_file_suffixes
        self.norm_stats = norm_stats
        self.max_len = max_len
        if command_list is not None:
            self.command_list = [cmd.strip("'\"") for cmd in command_list]
        self.total_items = 0
        self.use_language = use_language
        self.chunk_size = chunk_size
        # self.num_episodes = num_episodes
        self.task_config = task_config
        self.action_mode = task_config['action_mode'][0]
        self.norm_scheme = task_config['norm_scheme']

        self.camera_names = self.task_config['camera_names']
        self.camera_suffixes = self.task_config['camera_file_suffixes']

        if task_config.get('goal_condition_style'):
            self.goal_condition_style = task_config['goal_condition_style']
        else:
            self.goal_condition_style = None


        self.goal_circle_size = 10
        self.img_height, self.img_width = [360, 480]
        self.num_samples = task_config['num_episodes']
        
        self.tissue_phase_demo_dict = {}
        self.command_embeddings_dict = {}
        # self.command_embeddings_dict_json = {}

        for tissue_sample_id in tissue_sample_ids:
            tissue_sample_name = f"tissue_{tissue_sample_id}"
            tissue_sample_dir_path = os.path.join(dataset_dir, tissue_sample_name)
            phases = os.listdir(tissue_sample_dir_path)
            self.tissue_phase_demo_dict[tissue_sample_name] = {}

            for phase_sample in phases:
                demo_samples_path = os.path.join(tissue_sample_dir_path, phase_sample)

                if os.path.isfile(demo_samples_path):
                    continue  # Skip if the tissue sample path is not a directory

                demo_samples = os.listdir(demo_samples_path)

                ## remove corrections folder
                for demo_sample in demo_samples:
                    if demo_sample == "Corrections" or demo_sample.endswith(".json"):
                        demo_samples.remove(demo_sample)

                ## initialize the dictionary for the tissue sample
                if tissue_sample_name not in self.tissue_phase_demo_dict:
                    self.tissue_phase_demo_dict[tissue_sample_name] = {}

                # Add or update the demo samples in the dictionary
                self.tissue_phase_demo_dict[tissue_sample_name].setdefault(phase_sample, []).extend(demo_samples)


        print("num of tissues:", len(self.tissue_phase_demo_dict.keys()))
        print("phases:", self.tissue_phase_demo_dict[tissue_sample_name].keys())
        print("num of demos per phase:", {phase: len(demo_samples) for phase, demo_samples in self.tissue_phase_demo_dict[tissue_sample_name].items()})
        
        total_count = 0
        for phase_dict in self.tissue_phase_demo_dict.values():
            for demo_samples in phase_dict.values():
                total_count += len(demo_samples)
        self.num_samples = total_count
        print("total count:", total_count)
        ## create language embeddings
        if self.use_language:

            self.language_encoder = language_encoder
            # tokenizer, model = initialize_model_and_tokenizer(self.language_encoder)
            unique_phase_folder_names = np.unique([phase_folder_name for tissue_sample in self.tissue_phase_demo_dict.values() for phase_folder_name in tissue_sample.keys()])

            print("\ngenerating command embeddings...\n")

            json_name = f"candidate_embeddings_{self.language_encoder}.json"
            json_path = os.path.join(dataset_dir, json_name)

            self.command_embeddings_dict = self.get_command_embeddings_from_json(unique_phase_folder_names, json_path)
            print(self.command_embeddings_dict.keys())

        self.all_samples = [(tissue_sample, phase, sample) 
                            for tissue_sample in self.tissue_phase_demo_dict
                            for phase in self.tissue_phase_demo_dict[tissue_sample]
                            for sample in self.tissue_phase_demo_dict[tissue_sample][phase]]
        
        ## for weighted random sampler
        self.sample_task_labels = []
        for sample in self.all_samples:
            _, phase, _ = sample
            task_label = phase.split("_")[0]  # "1", "2", or "3"
            self.sample_task_labels.append(task_label)

        self.header_name_qpos_psm1 = ["psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
                                "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
                                "psm1_jaw"]
        
        self.header_name_qpos_psm2 = ["psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
                                "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
                                "psm2_jaw"]

        self.header_name_actions_psm1 = ["psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
                                    "psm1_sp.orientation.x", "psm1_sp.orientation.y", "psm1_sp.orientation.z", "psm1_sp.orientation.w",
                                    "psm1_jaw_sp"]

        self.header_name_actions_psm2 = ["psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
                                    "psm2_sp.orientation.x", "psm2_sp.orientation.y", "psm2_sp.orientation.z", "psm2_sp.orientation.w",
                                    "psm2_jaw_sp"]
        
        self.header_ecm = ["ecm_pose.position.x", "ecm_pose.position.y", "ecm_pose.position.z",
                            "ecm_pose.orientation.x", "ecm_pose.orientation.y", 
                            "ecm_pose.orientation.z", "ecm_pose.orientation.w"]
        
        self.quat_cp_psm1 = ["psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w"]
        self.quat_cp_psm2 = ["psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w"]

        self.transforms = DataAug([self.img_height, self.img_width])



    def get_command_embeddings_from_json(self, unique_phase_folder_names, json_file_name):
        phase_command_embeddings_dict = {}

        try:
            with open(json_file_name, "r") as f:
                episode_data = json.load(f)
        except FileNotFoundError:
            print(f"File {json_file_name} not found.")
            return phase_command_embeddings_dict
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file {json_file_name}.")
            return phase_command_embeddings_dict

        for phase_folder_name in tqdm(unique_phase_folder_names, desc="Embedding phase commands"):
            if phase_folder_name.endswith("_recovery"):
                phase_folder_name = phase_folder_name[:-9]
            
            _, phase_command = phase_folder_name.split("_")[0], " ".join(phase_folder_name.split("_")[1:])

            phase_command_embeddings_dict.setdefault(phase_folder_name, {})
            found_embedding = None
            for item in episode_data:
                if isinstance(item, dict) and item.get('command') == phase_command:
                    found_embedding = item.get('embedding')
                    break
        
            if found_embedding is not None:
                phase_command_embeddings_dict[phase_folder_name] = (phase_command, found_embedding)

            else:
                print(f"Embedding not found for command: {phase_command}")

        return phase_command_embeddings_dict

    
    def quat_to_axis_angle_action(self, action):
        """
        Convert a quaternion action to an axis-angle action.
        
        Args:
            action: Tensor of shape (..., 8) representing [x,y,z,qw,qx,qy,qz,jaw]

        Returns:
            axis_angle_actions: Tensor of shape (..., 7) representing [x,y,z,r11,r12,r13,r21,r22,r23,jaw]
        """
        quat_actions = action[:, 3:7]  # Shape: (n_actions, 4)

        r_actions = R.from_quat(quat_actions)
        diff_6d = r_actions.as_matrix()[:,:,:2]
        diff_6d = diff_6d.transpose(0,2,1).reshape(-1, 6) # first column then second column
        
        # Prepare the final diff array
        axis_angle_actions = np.zeros((action.shape[0], 10))  # Shape: (n_actions, 7)

        # Populate the diff_expand array
        axis_angle_actions[:, 0:3] = action[:, 0:3]     # Delta translation
        axis_angle_actions[:, 3:9] = diff_6d          # Delta rotation (axis-angle)
        axis_angle_actions[:, 9] = action[:, 7]         # Abs Jaw
        
        return axis_angle_actions

    # misnomer: jaw angles are also being normalized
    def min_max_scale_positions_only(self, diffs):
        """
        diffs: n_actions x 20
        return: normalized n_actions x 20
        Note: BOTH POSITIONS AND JAW ANGLES ARE NORMALIZED (orientations remain original)
        """
        max_ = self.task_config['action_mode'][1]['max_']
        min_ = self.task_config['action_mode'][1]['min_']
        normalized = (diffs - min_) / (max_ - min_) * 2 - 1

        # replace w/ originals for 6D rot
        normalized[:, 3:9] = diffs[:, 3:9]
        normalized[:, 13:19] = diffs[:, 13:19]

        return normalized
    
    def standardize_positions_only(self, diffs):
        """
        diffs: n_actions x 20
        return: normalized n_actions x 20 (zero mean unit variance)
        Note: BOTH POSITIONS AND JAW ANGLES ARE NORMALIZED (orientations remain original)
        """
        mean = self.task_config['action_mode'][1]['mean']
        std = self.task_config['action_mode'][1]['std']
        # print("mean shape", mean.shape)
        # print("std shape", std.shape)
        normalized = (diffs - mean) / std

        # replace w/ originals for 6D rot
        normalized[:, 3:9] = diffs[:, 3:9]
        normalized[:, 13:19] = diffs[:, 13:19]

        return normalized


    def preprocess_img(self, img, start_ts):
        if img is None:
            print("Image is None:", start_ts)
        img = cv2.resize(img, [self.img_width, self.img_height])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # construct observations
        img = torch.from_numpy(img).float() # channel last

        # bring channel to to the third
        img = torch.einsum('h w c -> c h w', img)

        # normalize image and change dtype to float
        img = img / 255.0

        return img

    def create_offset_map_with_gradient(self, image_shape, insert_point, exit_point, normalize_size=224.0, device='cpu', eps=1e-6):
        """
        Returns a 3-channel offset map:
        - Channel 0: dx to insertion point
        - Channel 1: dy to insertion point
        - Channel 2: scalar heatmap (1 at insertion, 0 at exit)

        Args:
            image_shape: (H, W)
            insert_point: (x, y)
            exit_point: (x, y)
            normalize_size: reference image size for normalization
            device: 'cpu' or 'cuda'
        """
        H, W = image_shape
        normalizing_constant = 250.0 * (min(H, W) / normalize_size)

        y_coords = torch.arange(H, device=device)
        x_coords = torch.arange(W, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Offsets to insertion point (dy, dx)
        dx = (x_grid - insert_point[0]) / normalizing_constant
        dy = (y_grid - insert_point[1]) / normalizing_constant

        # Gradient heatmap: insertion → 1.0, exit → 0.0
        d_insert = torch.sqrt((x_grid - insert_point[0]) ** 2 + (y_grid - insert_point[1]) ** 2)
        d_exit = torch.sqrt((x_grid - exit_point[0]) ** 2 + (y_grid - exit_point[1]) ** 2)
        heat = d_exit / (d_insert + d_exit + eps)  # in [0, 1]

        # Stack to shape (3, H, W)
        offset_map = torch.stack([dx, dy, heat], dim=0)
        return offset_map.clamp(-1.0, 1.0)  # Optional clamp


    def offset_map_to_rgb_visual(self, offset_map):
        """
        Converts a (3, H, W) offset map (dx, dy, heat) to a uint8 RGB image for visualization.
        - Red = dx
        - Green = dy
        - Blue = heat
        """
        if torch.is_tensor(offset_map):
            offset_map = offset_map.detach().cpu().numpy()

        # Normalize each channel to [0, 1]
        def normalize(x):
            x = x - np.min(x)
            x = x / (np.max(x) + 1e-6)
            return x

        dx_norm = normalize(offset_map[0])
        dy_norm = normalize(offset_map[1])
        heat_norm = normalize(offset_map[2])

        rgb_image = np.stack([
            dx_norm,     # R
            dy_norm,     # G
            heat_norm    # B
        ], axis=-1)  # (H, W, 3)

        rgb_uint8 = (rgb_image * 255).astype(np.uint8)
        return rgb_uint8

    def __len__(self):
     
        return len(self.episode_ids)


    def __getitem__(self, index):
        
        try:
            # Get the tissue sample, phase, and sample based on the index
            episode_id = self.episode_ids[index]
            if episode_id < self.num_samples:
                tissue_sample, phase, sample = self.all_samples[episode_id]
            else:
                print("episode_id out of range")
                tissue_sample, phase, sample = self.all_samples[episode_id % self.num_samples]

            dataset_path = os.path.join(self.dataset_dir, f"{tissue_sample}/{phase}/{sample}")
            csv_path = os.path.join(dataset_path, "ee_csv.csv")
            csv = pd.read_csv(csv_path)
            episode_len = len(csv)
            start_ts = np.random.choice(episode_len)

            img_idx = start_ts

            # -------------------------------
            # 1. Load raw image
            # -------------------------------        
            
            img_dict_raw = {}
            for cam_name, cam_suffix in zip(self.camera_names, self.camera_suffixes):
                subdir = {
                    '_left.jpg': 'left_img_dir',
                    '_right.jpg': 'right_img_dir',
                    '_psm1.jpg': 'endo_psm1',
                    '_psm2.jpg': 'endo_psm2',
                }.get(cam_suffix, 'left_img_dir')
                path = os.path.join(dataset_path, subdir, f"frame{img_idx:06d}{cam_suffix}")
                img = cv2.imread(path)
                if img is None:
                    raise FileNotFoundError(f"Image not found at: {path}")
                img_dict_raw[cam_name] = img

            # -------------------------------
            # 2. Plot goal points if needed (needle throw only)
            # -------------------------------
            if self.goal_condition_style == "plot" and phase.startswith("2_needle_throw"):
                # print("plotting goal points")
                clicked_csv_path = os.path.join(dataset_path, "clicked_point.csv")
                if os.path.exists(clicked_csv_path):
                    clicked = pd.read_csv(clicked_csv_path)
                    if not clicked.empty:
                        for _, row in clicked.iterrows():
                            x, y = int(row['x']), int(row['y'])
                            cv2.circle(img_dict_raw['left'], (x, y), self.goal_circle_size, (0, 255, 0), -1)

            if self.goal_condition_style == "mask":
                if phase.startswith("2_needle_throw"):
                    clicked_csv_path = os.path.join(dataset_path, "clicked_point.csv")
                    if os.path.exists(clicked_csv_path):
                        clicked = pd.read_csv(clicked_csv_path)
                        if not clicked.empty and len(clicked) >= 2:
                            # Create a 3-channel mask (H, W, 3) with all zeros
                            h, w = img_dict_raw["left"].shape[:2]
                            clicked_points_mask = np.zeros((h, w, 3), dtype=np.uint8)

                            # Draw insertion point (first point) as red
                            insert_x, insert_y = int(clicked.iloc[0]['x']), int(clicked.iloc[0]['y'])
                            cv2.circle(clicked_points_mask, (insert_x, insert_y), radius=10, color=(255, 0, 0), thickness=-1)  # Red in BGR

                            # Draw exit point (second point) as green
                            exit_x, exit_y = int(clicked.iloc[1]['x']), int(clicked.iloc[1]['y'])
                            cv2.circle(clicked_points_mask, (exit_x, exit_y), radius=10, color=(0, 255, 0), thickness=-1)  # Green in BGR

                            img_dict_raw["mask"] = clicked_points_mask
                        else:
                            print("clicked_point.csv has fewer than 2 points, skipping mask")
                            img_dict_raw["mask"] = np.zeros_like(img_dict_raw["left"])
                    else:
                        print("clicked_point.csv not found")
                        img_dict_raw["mask"] = np.zeros_like(img_dict_raw["left"])
                else:
                    img_dict_raw["mask"] = np.zeros_like(img_dict_raw["left"])

            if self.goal_condition_style == "dot" and phase.startswith("2_needle_throw"):
                # print("plotting goal points")
                clicked_csv_path = os.path.join(dataset_path, "clicked_point.csv")
                if os.path.exists(clicked_csv_path):
                    clicked_csv_path = os.path.join(dataset_path, "clicked_point.csv")
                    if os.path.exists(clicked_csv_path):
                        clicked = pd.read_csv(clicked_csv_path)
                        if not clicked.empty and len(clicked) >= 2:
                            # Create a 3-channel mask (H, W, 3) with all zeros
                            h, w = img_dict_raw["left"].shape[:2]
                            clicked_points_mask = np.zeros((h, w, 3), dtype=np.uint8)

                            # Draw insertion point (first point) as red
                            insert_x, insert_y = int(clicked.iloc[0]['x']), int(clicked.iloc[0]['y'])
                            cv2.circle(clicked_points_mask, (insert_x, insert_y), radius=10, color=(255, 0, 0), thickness=-1)  # Red in BGR

                            # Draw exit point (second point) as green
                            exit_x, exit_y = int(clicked.iloc[1]['x']), int(clicked.iloc[1]['y'])
                            cv2.circle(clicked_points_mask, (exit_x, exit_y), radius=10, color=(0, 255, 0), thickness=-1)  # Green in BGR

                            ## overlay the mask on the image
                            # Only blend where the mask has non-zero content
                            nonzero_mask = np.any(clicked_points_mask != 0, axis=-1)
                            overlay = img_dict_raw["left"].copy()
                            overlay[nonzero_mask] = cv2.addWeighted(
                                img_dict_raw["left"], 0.5, clicked_points_mask, 0.5, 0
                            )[nonzero_mask]
                            # overlay = cv2.addWeighted(img_dict_raw["left"], 0.5, clicked_points_mask, 0.5, 0)
                            img_dict_raw["left"] = overlay
                    else:
                        print("clicked_point.csv not found")

            if self.goal_condition_style == "map":
                if phase.startswith("2_needle_throw"):
                    # print("plotting goal points")
                    clicked_csv_path = os.path.join(dataset_path, "clicked_point.csv")
                    if os.path.exists(clicked_csv_path):
                        clicked = pd.read_csv(clicked_csv_path)
                        if not clicked.empty and len(clicked) >= 2:
                            # Create a 3-channel mask (H, W, 3) with all zeros
                            h, w = img_dict_raw["left"].shape[:2]

                            # Load clicked points
                            insert_x = int(clicked.iloc[0, 0])
                            insert_y = int(clicked.iloc[0, 1])
                            exit_x = int(clicked.iloc[1, 0])
                            exit_y = int(clicked.iloc[1, 1])
                            insert_point = (insert_x, insert_y)
                            exit_point = (exit_x, exit_y)
                            # Create offset map
                            offset_map = self.create_offset_map_with_gradient(
                                image_shape=(h, w),
                                insert_point=insert_point,
                                exit_point=exit_point,
                                device='cpu'
                            )

                            rgb_offset_viz = self.offset_map_to_rgb_visual(offset_map)
                            # img_dict_raw["left"] = cv2.addWeighted(img_dict_raw["left"], 0.5, rgb_offset_viz, 0.5, 0)
                            img_dict_raw["mask"] = rgb_offset_viz
                        else:
                            print("clicked_point.csv has fewer than 2 points, skipping mask")
                            img_dict_raw["mask"] = np.zeros_like(img_dict_raw["left"])
                    else:
                        print("clicked_point.csv not found")
                        img_dict_raw["mask"] = np.zeros_like(img_dict_raw["left"])
                else:
                    img_dict_raw["mask"] = np.zeros_like(img_dict_raw["left"])

            # -------------------------------
            # 3. Preprocess and augment images
            # -------------------------------
            img_dict = {k: self.preprocess_img(v, start_ts) for k, v in img_dict_raw.items()}


            # -------------------------------
            #  4. Apply data augmentation
            # -------------------------------

            tfmed = self.transforms(img_dict)
            image_data = np.stack([tfmed[k] for k in sorted(tfmed.keys())], axis=0)

            # -------------------------------
            #  5. Load and compute action data
            # -------------------------------
            action_psm1 = csv[self.header_name_actions_psm1].iloc[start_ts:start_ts+self.chunk_size].to_numpy() # note 400 added here
            action_psm2 = csv[self.header_name_actions_psm2].iloc[start_ts:start_ts+self.chunk_size].to_numpy() # note 400 added here

            if self.action_mode == 'hybrid':
                # convert to axis-angle actions
                axis_angle_actions_psm1 = self.quat_to_axis_angle_action(action_psm1)
                axis_angle_actions_psm2 = self.quat_to_axis_angle_action(action_psm2)

                diff_psm1 = np.zeros((self.chunk_size, 10))
                diff_psm2 = np.zeros((self.chunk_size, 10))

                # Pad the actions up to the action horizon
                diff_psm1[:axis_angle_actions_psm1.shape[0], :] = axis_angle_actions_psm1
                diff_psm2[:axis_angle_actions_psm2.shape[0], :] = axis_angle_actions_psm2
            else:
                raise(NotImplementedError) 

            # stack the actions along column dim
            action = np.column_stack((diff_psm1, diff_psm2))

            # normalize data
            if self.norm_scheme == 'min_max': 
                action = self.min_max_scale_positions_only(action)
            elif self.norm_scheme == 'std':
                action = self.standardize_positions_only(action)
            else:
                raise NotImplementedError

            action_len = min(episode_len - start_ts, self.chunk_size)
            padded_action = np.zeros((self.chunk_size, 20), dtype=np.float32) 
            padded_action[:action_len] = action[:action_len]
            is_pad = np.zeros(self.chunk_size)
            is_pad[action_len:] = 1

            qpos = np.zeros(20)

            # construct observations
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()

            # -------------------------------
            #  7. Command Embedding
            # -------------------------------
            if self.use_language:
                if phase.endswith("_recovery"):
                    phase = phase[:-9]

                command_tuple = self.command_embeddings_dict[phase]

                _, embedding = command_tuple
                command_embedding = torch.tensor(embedding).squeeze()

                return image_data, qpos_data, action_data, is_pad, command_embedding

            return image_data, qpos_data, action_data, is_pad
        
        # Handle exceptions
        except FileNotFoundError as e:
            print(f"File not found at index {index}: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            print(f"Empty data error at index {index}: {e}")
            raise
        except KeyError as e:
            print(f"Key error at index {index}: {e}")
            raise
        except ValueError as e:
            print(f"Value error at index {index}: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error at index {index}: {e}")                



        
"""
Test the EpisodicDatasetDvrkGeneric class.
"""
if __name__ == "__main__":
    set_seed(0)
    for i in range(10):

        path_to_dataset = os.getenv("PATH_TO_DATASET")
        dataset_dir = os.path.join(path_to_dataset)
        use_language_flag = True
        from dvrk_scripts.constants_dvrk import TASK_CONFIGS
        task_config = TASK_CONFIGS['suturing_dot']
        camera_names = task_config['camera_names']
        tissue_samples_ids = task_config["tissue_samples_ids"]
        num_episodes = task_config["num_episodes"]
        camera_file_suffixes = task_config['camera_file_suffixes']
        episode_ids = [i for i in range(num_episodes)]
        dataset = EpisodicDatasetDvrkGeneric(
                    episode_ids,
                    tissue_samples_ids,
                    dataset_dir,
                    camera_names,
                    camera_file_suffixes,
                    task_config,
                    chunk_size=60,
                    use_language=use_language_flag
                    )

        # Sample a random item from the dataset
        rdm_idx = np.random.randint(0, len(dataset))
        print("idx:", rdm_idx)
        if use_language_flag:
            image_data, qpos_data, action_data, is_pad, command_embedding = dataset[rdm_idx]
        else:
            image_data, qpos_data, action_data, is_pad = dataset[rdm_idx]   


        # Create a figure with subplots: one row per timestamp, one column per camera
        fig, axes = plt.subplots(1, len(image_data), figsize=(15, 10))
        for cam_idx, img in enumerate(image_data):

            # Check and possibly transpose the shape if needed
            if img.shape[0] == 3 and len(img.shape) == 3:
                img = np.transpose(img, (1, 2, 0))  # Transpose to (height, width, channels)

            axes[cam_idx].imshow(img)
            axes[cam_idx].axis('off')  # Optionally turn off the axis


        plt.savefig(f"./visualization_{i}.png")
        