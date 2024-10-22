from __future__ import absolute_import, division, print_function
import logging

import tqdm
import json
import numpy as np
from dataset.JointsDataset import JointsDataset
from skelda import evals, utils_pose

logger = logging.getLogger(__name__)


# ==================================================================================================

dataset_use = "human36m"
# dataset_use = "panoptic"
# dataset_use = "mvor"
# dataset_use = "shelf"
# dataset_use = "campus"
# dataset_use = "ikeaasm"
# dataset_use = "tsinghua"
datasets = {
    "panoptic": {
        "path": "/datasets/panoptic/skelda/test.json",
        "cams": ["00_03", "00_06", "00_12", "00_13", "00_23"],
        "take_interval": 3,
        "use_scenes": ["160906_pizza1", "160422_haggling1", "160906_ian5"],
    },
    "human36m": {
        "path": "/datasets/human36m/skelda/pose_test.json",
        "take_interval": 5,
    },
    "mvor": {
        "path": "/datasets/mvor/skelda/all.json",
        "take_interval": 1,
    },
    "ikeaasm": {
        "path": "/datasets/ikeaasm/skelda/test.json",
        "take_interval": 2,
    },
    "campus": {
        "path": "/datasets/campus/skelda/test.json",
        "take_interval": 1,
    },
    "shelf": {
        "path": "/datasets/shelf/skelda/test.json",
        "take_interval": 1,
    },
    "tsinghua": {
        "path": "/datasets/tsinghua/skelda/test.json",
        "take_interval": 3,
    },
}

joint_names_3d = [
    "shoulder_middle",
    "nose",
    "hip_middle",
    "shoulder_left",
    "elbow_left",
    "wrist_left",
    "hip_left",
    "knee_left",
    "ankle_left",
    "shoulder_right",
    "elbow_right",
    "wrist_right",
    "hip_right",
    "knee_right",
    "ankle_right",
]

eval_joints = [
    "nose",
    "shoulder_left",
    "shoulder_right",
    "elbow_left",
    "elbow_right",
    "wrist_left",
    "wrist_right",
    "hip_left",
    "hip_right",
    "knee_left",
    "knee_right",
    "ankle_left",
    "ankle_right",
]

# ==================================================================================================


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


# ==================================================================================================


def load_labels(dataset: dict):
    """Load labels by dataset description"""

    if "panoptic" in dataset:
        labels = load_json(dataset["panoptic"]["path"])
        labels = [lb for i, lb in enumerate(labels) if i % 1500 < 90]

        # Filter by maximum number of persons
        labels = [l for l in labels if len(l["bodies3D"]) <= 10]

        # Filter scenes
        if "use_scenes" in dataset["panoptic"]:
            labels = [
                l for l in labels if l["scene"] in dataset["panoptic"]["use_scenes"]
            ]

        # Filter cameras
        if not "cameras_depth" in labels[0]:
            for label in labels:
                for i, cam in reversed(list(enumerate(label["cameras"]))):
                    if cam["name"] not in dataset["panoptic"]["cams"]:
                        label["cameras"].pop(i)
                        label["imgpaths"].pop(i)

    elif "human36m" in dataset:
        labels = load_json(dataset["human36m"]["path"])
        labels = [lb for lb in labels if lb["subject"] == "S9"]
        labels = [lb for i, lb in enumerate(labels) if i % 4000 < 150]

        for label in labels:
            label.pop("action")
            label.pop("frame")

    elif "mvor" in dataset:
        labels = load_json(dataset["mvor"]["path"])

        # Rename keys
        for label in labels:
            label["cameras_color"] = label["cameras"]
            label["imgpaths_color"] = label["imgpaths"]

            # Use "head" label for "nose" detections
            label["joints"][label["joints"].index("head")] = "nose"

    elif "ikeaasm" in dataset:
        labels = load_json(dataset["ikeaasm"]["path"])
        cams0 = str(labels[0]["cameras"])
        labels = [lb for lb in labels if str(lb["cameras"]) == cams0]

    elif "shelf" in dataset:
        labels = load_json(dataset["shelf"]["path"])
        labels = [lb for lb in labels if "test" in lb["splits"]]

        # Use "head" label for "nose" detections
        for label in labels:
            label["joints"][label["joints"].index("head")] = "nose"

    elif "campus" in dataset:
        labels = load_json(dataset["campus"]["path"])
        labels = [lb for lb in labels if "test" in lb["splits"]]

        # Use "head" label for "nose" detections
        for label in labels:
            label["joints"][label["joints"].index("head")] = "nose"

    elif "tsinghua" in dataset:
        labels = load_json(dataset["tsinghua"]["path"])
        labels = [lb for lb in labels if "test" in lb["splits"]]
        labels = [lb for lb in labels if lb["seq"] == "seq_1"]
        labels = [lb for i, lb in enumerate(labels) if i % 300 < 90]

        for label in labels:
            label["bodyids"] = list(range(len(label["bodies3D"])))

    else:
        raise ValueError("Dataset not available")

    # Optionally drop samples to speed up train/eval
    if "take_interval" in dataset:
        take_interval = dataset["take_interval"]
        if take_interval > 1:
            labels = [l for i, l in enumerate(labels) if i % take_interval == 0]

    # Filter joints
    fj_func = lambda x: utils_pose.filter_joints_3d(x, joint_names_3d)
    labels = list(map(fj_func, labels))

    return labels


# ==================================================================================================


class Skelda(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, is_train, transform)
        
        self.num_joints = len(joint_names_3d)
        self.num_views = cfg.DATASET.CAMERA_NUM
        self.root_id = cfg.DATASET.ROOTIDX

        self.has_evaluate_function = True
        self.transform = transform

        print("Loading labels ...")
        if is_train == "train":
            labels = []

        else:
            ds = datasets[dataset_use]
            labels = load_labels(
                {dataset_use: ds, "take_interval": ds["take_interval"]}
            )
        self.labels = labels

        # Print a dataset sample for debugging
        print(labels[0])

        self.has_views = len(labels[0]["cameras"])
        self.num_views = cfg.DATASET.CAMERA_NUM
        print(self.num_views, self.has_views)

        self._get_db()
        self.db_size = len(self.db)
        print(len(self.labels), self.db_size)

    def _get_db(self):
        db = []

        for label in tqdm.tqdm(self.labels):
            
            for i in range(len(label["cameras"])):
                cam = label["cameras"][i]

                all_poses_3d = []
                all_poses_vis_3d = []
                all_poses = []
                all_poses_vis = []

                for body in label["bodies3D"]:
                    pose = np.array(body)

                    pose3d = pose[:, 0:3] * 1000
                    vis3d = pose[:, -1] > 0.1

                    all_poses_3d.append(pose3d)
                    all_poses_vis_3d.append(
                        np.repeat(np.reshape(vis3d, (-1, 1)), 3, axis=1)
                    )

                    pose2d, dist = utils_pose.project_poses(
                        np.expand_dims(pose, axis=0), cam
                    )
                    pose2d, dist = pose2d[0], dist[0]
                    vis2d = pose2d[:, 2] > 0
                    vis2d = vis2d * vis3d
                    pose2d = pose2d[:, 0:2]

                    all_poses.append(pose2d)
                    all_poses_vis.append(
                        np.repeat(np.reshape(vis2d, (-1, 1)), 2, axis=1)
                    )

                # if len(all_poses_3d) == 0:
                #     continue

                our_cam = {}
                our_cam["R"] = np.array(cam["R"])
                our_cam["T"] = np.array(cam["T"]) * 1000
                our_cam["fx"] = np.array(cam["K"])[0, 0]
                our_cam["fy"] = np.array(cam["K"])[1, 1]
                our_cam["cx"] = np.array(cam["K"])[0, 2]
                our_cam["cy"] = np.array(cam["K"])[1, 2]
                our_cam["k"] = np.array(cam["DC"])[[0, 1, 4]].reshape(3, 1)
                our_cam["p"] = np.array(cam["DC"])[[2, 3]].reshape(2, 1)

                item = {
                    "key": label["id"],
                    "image": label["imgpaths"][i],
                    "idx": label["id"],
                    "joints_3d": all_poses_3d,
                    "joints_3d_vis": all_poses_vis_3d,
                    "joints_2d": all_poses,
                    "joints_2d_vis": all_poses_vis,
                    "camera": our_cam,
                }
                db.append(item)

        self.db = db
        return


    def __getitem__(self, idx):
        input, target, weight, target_3d, meta, input_heatmap = [], [], [], [], [], []

        # if self.image_set == 'train':
        #     # camera_num = np.random.choice([5], size=1)
        #     select_cam = np.random.choice(self.num_views, size=5, replace=False)
        # elif self.image_set == 'validation':
        #     select_cam = list(range(self.num_views))

        for k in range(self.num_views):
            i, t, w, t3, m, ih = super().__getitem__(self.num_views * idx + k)
            if i is None:
                continue
            input.append(i)
            target.append(t)
            weight.append(w)
            target_3d.append(t3)
            meta.append(m)
            input_heatmap.append(ih)
        return input, target, weight, target_3d, meta, input_heatmap

    def __len__(self):
        return self.db_size // self.num_views

    # ==============================================================================================

    def add_extra_joints(self, poses3D):
        # Add "hip_middle" joint
        hip_left = poses3D[:, joint_names_3d.index("hip_left"), :]
        hip_right = poses3D[:, joint_names_3d.index("hip_right"), :]
        hip_middle = (hip_left[:, 0:3] + hip_right[:, 0:3]) / 2
        hip_middle = np.concatenate(
            (hip_middle, np.minimum(hip_left[:, 3], hip_right[:, 3])[:, np.newaxis]),
            axis=-1,
        )
        poses3D = np.concatenate((poses3D, hip_middle[:, np.newaxis, :]), axis=-2)

        # Add "shoulder_middle" joint
        shoulder_left = poses3D[:, joint_names_3d.index("shoulder_left"), :]
        shoulder_right = poses3D[:, joint_names_3d.index("shoulder_right"), :]
        shoulder_middle = (shoulder_left[:, 0:3] + shoulder_right[:, 0:3]) / 2
        shoulder_middle = np.concatenate(
            (
                shoulder_middle,
                np.minimum(shoulder_left[:, 3], shoulder_right[:, 3])[:, np.newaxis],
            ),
            axis=-1,
        )
        poses3D = np.concatenate((poses3D, shoulder_middle[:, np.newaxis, :]), axis=-2)

        return poses3D

    # ==============================================================================================

    def evaluate(self, preds):
        global joint_names_3d

        all_poses = preds
        all_ids = [r["id"] for r in self.labels]

        filtered_poses = []
        scale = [1000.0, 1000.0, 1000.0, 1.0]
        for poses in all_poses:

            new_poses = []
            for pose in poses:
                if pose[0, 3] >= 0:
                    pose = np.array(pose)[:, [0, 1, 2, 4]] / scale
                    new_poses.append(pose)
            filtered_poses.append(new_poses)
        all_poses = filtered_poses

        # from skelda import utils_view
        # for i, poses3D in enumerate(all_poses):
        #     camparams = self.labels[i]["cameras"]
        #     roomparams = {
        #         "room_size": self.labels[i]["room_size"],
        #         "room_center": self.labels[i]["room_center"],
        #     }
        #     _ =  utils_view.show_poses3d(
        #         poses3D, joint_names_3d, roomparams, camparams
        #     )
        #     _ = utils_view.show_poses3d(
        #         self.labels[i]["bodies3D"], eval_joints, roomparams, camparams
        #     )
        #     utils_view.show_plots()

        res = evals.mpjpe.run_eval(
            self.labels,
            all_poses,
            all_ids,
            joint_names_net=joint_names_3d,
            joint_names_use=eval_joints,
            save_error_imgs="",
        )
        _ = evals.pcp.run_eval(
            self.labels,
            all_poses,
            all_ids,
            joint_names_net=joint_names_3d,
            joint_names_use=eval_joints,
            replace_head_with_nose=True,
        )

        if "mpjpe" in res:
            metric = [v for k,v in res["mpjpe"].items() if k.startswith("ap-")]
        else:
            metric = [0]

        return metric, [], [], []
