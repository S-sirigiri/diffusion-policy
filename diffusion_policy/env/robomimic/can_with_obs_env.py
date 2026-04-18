import copy

from robosuite.environments.base import register_env
from robosuite.environments.manipulation.pick_place import PickPlaceCan
from robosuite.models.objects import CylinderObject
from robosuite.utils.mjcf_utils import array_to_string


DEFAULT_OBSTACLE_CFG = {
    "pos": [0.00, -0.10, 0.07],
    "radius": 0.03,
    "height": 0.14,
    "rgba": [0.35, 0.35, 0.35, 1.0],
}


def _merge_obstacle_cfg(obstacle_cfg=None):
    merged = copy.deepcopy(DEFAULT_OBSTACLE_CFG)
    if obstacle_cfg is not None:
        merged.update(copy.deepcopy(obstacle_cfg))
    return merged


class CanWithObs(PickPlaceCan):
    def __init__(self, obstacle_cfg=None, **kwargs):
        self.obstacle_cfg = _merge_obstacle_cfg(obstacle_cfg)
        self.obstacle = None
        self.obstacle_body = None
        super().__init__(**kwargs)

    def _load_model(self):
        super()._load_model()

        obstacle_cfg = self.obstacle_cfg
        obstacle = CylinderObject(
            name="can_obstacle",
            size=[obstacle_cfg["radius"], obstacle_cfg["height"] / 2.0],
            rgba=obstacle_cfg["rgba"],
            joints=None,
            obj_type="all",
        )
        obstacle_body = obstacle.get_obj()
        obstacle_body.set("pos", array_to_string(obstacle_cfg["pos"]))

        self.model.merge_assets(obstacle)
        bin1_body = self.model.worldbody.find("./body[@name='bin1']")
        if bin1_body is None:
            raise RuntimeError("Failed to locate bin1 body while adding can obstacle.")
        bin1_body.append(obstacle_body)

        self.obstacle = obstacle
        self.obstacle_body = obstacle.root_body


register_env(CanWithObs)


def patch_can_with_obs_env_meta(env_meta, obstacle_cfg=None):
    patched = copy.deepcopy(env_meta)
    patched["env_name"] = "CanWithObs"
    patched["env_kwargs"].pop("single_object_mode", None)
    patched["env_kwargs"].pop("object_type", None)
    patched["env_kwargs"]["obstacle_cfg"] = _merge_obstacle_cfg(obstacle_cfg)
    return patched
