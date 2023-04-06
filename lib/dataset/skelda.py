from __future__ import absolute_import, division, print_function

import copy
import logging
import os

import tqdm
import json_tricks as json
import numpy as np
from dataset.JointsDataset import JointsDataset
from skelda import readers, utils_pose

logger = logging.getLogger(__name__)


# ==================================================================================================

# joint_names_3d = [
#     "shoulder_middle",
#     "nose",
#     "hip_middle",
#     "shoulder_left",
#     "elbow_left",
#     "wrist_left",
#     "hip_left",
#     "knee_left",
#     "ankle_left",
#     "shoulder_right",
#     "elbow_right",
#     "wrist_right",
#     "hip_right",
#     "knee_right",
#     "ankle_right",
# ]

joint_names_3d = [
    "ankle_right",
    "knee_right",
    "hip_right",
    "hip_left",
    "knee_left",
    "ankle_left",
    "hip_middle",
    "spine_upper",
    "shoulder_middle",
    "head",
    "wrist_right",
    "elbow_right",
    "shoulder_right",
    "shoulder_left",
    "elbow_left",
    "wrist_left",
    "nose",
]

dataset_use_train = "human36m"
dataset_use_eval = "human36m"
datasets = {
    "train": {
        "panoptic": {
            "path": "/datasets/panoptic/train/",
            "cams": ["00_03", "00_06", "00_12", "00_13", "00_23"],
            # "cams": ["00_15", "00_10", "00_21", "00_09", "00_01"],
            "take_interval": 120,
            # "use_scenes": ["160906_pizza1", "160422_haggling1", "160906_ian5"],
            # "kinect_cams": [e - 1 for e in [2, 4, 5, 6, 8]],
        },
        "human36m": {
            "path": "/datasets/human36m/",
            "subjects": ["S1", "S5", "S6", "S7", "S8"],
            "take_interval": 10,
        },
    },
    "eval": {
        "panoptic": {
            "path": "/datasets/panoptic/depth_test/",
            "cams": ["00_03", "00_06", "00_12", "00_13", "00_23"],
            # "cams": ["00_15", "00_10", "00_21", "00_09", "00_01"],
            "take_interval": 120,
            # "use_scenes": ["160906_pizza1", "160422_haggling1", "160906_ian5"],
            "kinect_cams": [e - 1 for e in [2, 4, 5, 6, 8]],
        },
        "human36m": {
            "path": "/datasets/human36m/",
            "subjects": ["S9", "S11"],
            "take_interval": 100,
        },
    },
}

reset_cache = True
cachefile = "/VoxelPoseWrapper/data/cache.json"

# ==================================================================================================


def load_labels(dataset: dict):
    """Load labels by dataset description"""

    if os.path.exists(cachefile) and not reset_cache:
        with open(cachefile, "r", encoding="utf-8") as file:
            labels = json.load(file)
    else:
        if "panoptic" in dataset:
            # labels = readers.panoptic.ds_multiview(
            labels = readers.panoptic_depth.ds_multiview_depth(
                datapath=dataset["panoptic"]["path"],
                joints=joint_names_3d,
            )

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
            else:
                new_labels = []
                for label in labels:
                    # Replace default cameras with kinect cameras
                    label["imgpaths"] = [
                        v
                        for i, v in enumerate(label["imgpaths_color"])
                        if i in dataset["panoptic"]["kinect_cams"]
                    ]
                    label["cameras"] = [
                        v
                        for i, v in enumerate(label["cameras_color"])
                        if i in dataset["panoptic"]["kinect_cams"]
                    ]
                    if not any([ip == "" for ip in label["imgpaths"]]):
                        new_labels.append(label)
                labels = new_labels

        elif "human36m" in dataset:
            labels = readers.human36m.ds_multiview(
                datapath=dataset["human36m"]["path"],
                subjects=dataset["human36m"]["subjects"],
                joints=joint_names_3d,
            )

            for label in labels:
                label.pop("action")
                label.pop("frame")

        else:
            raise ValueError("Dataset not available")

        # Save labels to cache
        with open(cachefile, "w+", encoding="utf-8") as file:
            json.dump(labels, file, indent=2)

    # Optionally drop samples to speed up train/eval
    if "take_interval" in dataset:
        take_interval = dataset["take_interval"]
        if take_interval > 1:
            labels = [l for i, l in enumerate(labels) if i % take_interval == 0]

    return labels


# ==================================================================================================


class Skelda(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)

        self.joints_def = joint_names_3d
        self.num_joints = len(joint_names_3d)

        if self.image_set == "train":
            ds = datasets["train"][dataset_use_train]
            labels = load_labels(
                {dataset_use_train: ds, "take_interval": ds["take_interval"]}
            )

        elif self.image_set == "validation":
            ds = datasets["eval"][dataset_use_eval]
            labels = load_labels(
                {dataset_use_eval: ds, "take_interval": ds["take_interval"]}
            )
        self.labels = labels

        # Print a dataset sample for debugging
        print(labels[0])

        self.num_views = len(labels[0]["cameras"])
        self.db = self._get_db()
        self.db_size = len(self.db)

    def _get_db(self):
        db = []

        for label in tqdm.tqdm(self.labels):
            for i in range(len(label["cameras"])):
                cam = label["cameras"][i]

                all_poses_3d = []
                all_poses_vis_3d = []
                all_poses = []
                all_poses_vis = []
                all_poses_dists = []

                for body in label["bodies3D"]:
                    pose = np.array(body)
                    pose3d = pose[:, 0:3]
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
                    all_poses_dists.append(dist)
                    all_poses_vis.append(
                        np.repeat(np.reshape(vis2d, (-1, 1)), 2, axis=1)
                    )

                if len(all_poses_3d) > 0:
                    our_cam = {}
                    our_cam["R"] = np.array(cam["R"])
                    our_cam["T"] = -1 * np.array(cam["T"])
                    our_cam["fx"] = np.array(cam["K"])[0, 0]
                    our_cam["fy"] = np.array(cam["K"])[1, 1]
                    our_cam["cx"] = np.array(cam["K"])[0, 2]
                    our_cam["cy"] = np.array(cam["K"])[1, 2]
                    our_cam["k"] = np.array(cam["DC"])[[0, 1, 4]].reshape(3, 1)
                    our_cam["p"] = np.array(cam["DC"])[[2, 3]].reshape(2, 1)

                    item = {
                        "key": "{}_{}{}".format(
                            label["scene"],
                            cam["name"],
                            os.path.basename(label["imgpaths"][i])
                            .replace(cam["name"], "")
                            .split(".")[0],
                        ),
                        "image": label["imgpaths"][i],
                        "joints_3d": all_poses_3d,
                        "joints_3d_vis": all_poses_vis_3d,
                        "joints_2d": all_poses,
                        "joints_2d_dists": all_poses_dists,
                        "joints_2d_vis": all_poses_vis,
                        "camera": our_cam,
                    }

                    db.append(item)
        return db

    def __getitem__(self, idx):
        input, target, weight, target_3d, meta, input_heatmap = [], [], [], [], [], []

        select_cams = list(range(self.num_views))
        if self.image_set == "train" and self.num_views > 5:
            select_cams = np.random.choice(self.num_views, size=5, replace=False)

        for k in select_cams:
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

    def evaluate(self, preds):
        eval_list = []
        gt_num = self.db_size // self.num_views
        assert len(preds) == gt_num, "number mismatch"

        total_gt = 0
        for i in range(gt_num):
            index = self.num_views * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec["joints_3d"]
            joints_3d_vis = db_rec["joints_3d_vis"]

            if len(joints_3d) == 0:
                continue

            pred = preds[i].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                mpjpes = []
                for gt, gt_vis in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(
                        np.sqrt(np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1))
                    )
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append(
                    {
                        "mpjpe": float(min_mpjpe),
                        "score": float(score),
                        "gt_id": int(total_gt + min_gt),
                    }
                )

            total_gt += len(joints_3d)

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        return (
            aps,
            recs,
            self._eval_list_to_mpjpe(eval_list),
            self._eval_list_to_recall(eval_list, total_gt),
        )

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

        return len(np.unique(gt_ids)) / total_gt
