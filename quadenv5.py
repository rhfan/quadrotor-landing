import torch
import genesis as gs
import math
import genesis.utils.geom as geom
import numpy as np


# 1、到达目标条件更为约束
# 2、reward追加速度约束
# 3、更严格的终止条件
# 4、目标位置更新及环境重启代码结构重构
# 5\bug修复：目标位置更新错误
# 6\bug修复：目标位置设置调为x,y,0.15
class QuadEnv_polyv3:
    def __init__(self, num_envs: int, show_viewer: bool = False):
        # 1、初始化genesis场景
        # 1.1、并行环境设置
        self.num_envs = num_envs  # 并行环境数
        self.rendered_env_num = min(10, self.num_envs)  # 展示环境数
        # 1.2、环境参数设置
        self.device = gs.device
        self.simulate_action_latency = True  # 仿真动作延迟
        self.dt = 0.01  # 仿真时间步长
        self.episode_length = 15  # 环境最大运行时长15
        self.max_episode_length = math.ceil(self.episode_length / self.dt)
        # 1.3、创建场景
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(max_FPS=60),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(self.rendered_env_num))
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        # 1.4、添加地面
        self.scene.add_entity(gs.morphs.Plane())
        # 1.5、增加目标
        self.target = self.scene.add_entity(
            morph=gs.morphs.MJCF(file="xml/car.xml", scale=0.5)
        )
        # 1.6、增加无人机
        self.drone = self.scene.add_entity(
            gs.morphs.Drone(file="urdf/drones/cf2x.urdf")
        )
        # 1.7、添加视角
        self.cam = self.scene.add_camera(
            res=(640, 480),
            pos=(3.5, 0.0, 2.5),
            lookat=(0, 0, 0.5),
            fov=30,
            GUI=True,
        )  # 用于展示
        # 1.8、build scene
        self.scene.build(n_envs=self.num_envs)
        # 2、初始化环境变量
        # 2.1、无人机初始位置
        self.base_init_pos = torch.tensor(
            [0.0, 0.0, 1.0], device=gs.device
        )  # 从地面高度1.5m出发
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        self.inv_base_init_quat = geom.inv_quat(self.base_init_quat)
        # 2.2、环境状态变量
        self.num_obs = (
            17  # 位置（3）、线速度（3）、角度四元数（4）、角速度（3）、动作（4）
        )
        self.num_actions = 4
        self.num_commands = 3  # 位置（3）
        self.num_privileged_obs = None
        # 2.3奖励设置
        self.reward_scales = {
            "target": 10.0,
            "smooth": -1e-3,
            "crash": -10.0,
        }
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )
        # 2.5、环境缓冲区
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float
        )  # 观测值缓冲区
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )  # 奖励缓冲区
        self.reset_buf = torch.ones(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )  # 重置缓冲区，初始全1保证开始环境初始化
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )  # 每个环境的当前episode长度
        self.target_pos_buf = torch.zeros(
            (self.num_envs, self.max_episode_length, self.num_commands),
            device=gs.device,
            dtype=gs.tc_float,
        )  # 目标位置缓冲区
        # self.target_pos=torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)#当前目标位置
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float
        )  # 动作缓冲区
        self.last_actions = torch.zeros_like(self.actions)  # 上一次运行动作
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.last_base_pos = torch.zeros_like(self.base_pos)
        # 日志记录
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    # 运行所需函数
    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    # 目标轨迹调取
    def target_move_idx(self, env_idx):
        # 生成随机数
        num = np.random.randint(0, 5000)
        name = "./mobile_data_poly/" + str(num) + ".npy"
        data = np.load(name)
        data = data.reshape(-1, 2)
        z = np.full(
            (data.shape[0], 1), 0.20
        )  # 小车几何中心0.15，小车表面厚度0.05，因此无人机目标位置高度为0.2
        data = np.concatenate((data, z), axis=1)
        self.target_pos_buf[env_idx] = torch.as_tensor(
            data, device=gs.device, dtype=gs.tc_float
        )
        # self.target_pos[env_idx]=self.target_pos_buf[env_idx,0]

    # 环境重置
    def reset_idx(self, env_idx):
        # 重置无人机位置、四元数信息
        self.base_pos[env_idx] = self.base_init_pos
        self.last_base_pos[env_idx] = self.base_init_pos
        self.base_quat[env_idx] = self.base_init_quat
        # 重置目标位置
        self.target_move_idx(env_idx)
        # 更新无人机位置
        self.drone.set_pos(self.base_pos[env_idx], zero_velocity=True, envs_idx=env_idx)
        self.drone.set_quat(
            self.base_quat[env_idx], zero_velocity=True, envs_idx=env_idx
        )
        self.base_lin_vel[env_idx] = 0.0
        self.base_ang_vel[env_idx] = 0.0
        self.drone.zero_all_dofs_velocity(env_idx)
        # 重置缓冲区
        self.last_actions[env_idx] = 0.0
        self.episode_length_buf[env_idx] = 0
        self.reset_buf[env_idx] = True
        # 日志填充
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_idx]).item() / self.episode_length
            )

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # 更新与目标位置
        self.rel_pos = self.target_pos_buf[:, 0, :] - self.base_pos
        self.last_rel_pos = self.target_pos_buf[:, 0, :] - self.last_base_pos
        return self.obs_buf

    # 目标到达
    def at_target(self):
        # 目标到达条件；1、与地面距离小于0.15；2、与目标距离小于0.05.训练时可逐步放宽
        cond = (torch.norm(self.rel_pos[:, :2], dim=1) < 0.1) & (
            torch.abs(self.rel_pos[:, 2]) < 0.03
        )
        # cond=torch.norm(self.rel_pos, dim=1) < 0.1
        return cond

    # 步骤递进
    def step(self, actions):
        # 动作提取
        clip_actions = 1
        self.actions = torch.clip(actions, -clip_actions, clip_actions)
        exec_actions = self.actions
        # 14468 is hover rpm
        self.drone.set_propellels_rpm((1 + exec_actions * 0.8) * 14468.429183500699)
        # 更新目标位置
        # 越界直接重置
        # target_pos = self.target_pos_buf[torch.arange(self.target_pos_buf.shape[0]), self.episode_length_buf, :]
        target_pos = self.target_pos_buf[
            torch.arange(self.target_pos_buf.shape[0]),
            self.episode_length_buf % self.max_episode_length,
            :,
        ]
        self.target.set_pos(target_pos, zero_velocity=True)
        self.scene.step()
        # 更新buffer
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()

        self.rel_pos = target_pos - self.base_pos
        self.last_rel_pos = target_pos - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = geom.quat_to_xyz(
            geom.transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            ),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = geom.inv_quat(self.base_quat)
        self.base_lin_vel[:] = geom.transform_by_quat(
            self.drone.get_vel(), inv_base_quat
        )
        self.base_ang_vel[:] = geom.transform_by_quat(
            self.drone.get_ang(), inv_base_quat
        )
        # 检查终止和重置环境，相较于v2放宽坠毁条件
        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > 75)
            | (torch.abs(self.base_euler[:, 0]) > 75)
            | (torch.abs(self.rel_pos[:, 0]) > 3.0)
            | (torch.abs(self.rel_pos[:, 1]) > 3.0)
            | (torch.abs(self.rel_pos[:, 2]) > 2.0)
            | (self.base_pos[:, 2] < 0.1)
        )
        self.reset_buf = (
            (self.episode_length_buf > self.max_episode_length)
            | self.crash_condition
            | self.at_target()
        )
        time_out_idx = (
            (self.episode_length_buf > self.max_episode_length)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=gs.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0
        reset_env = self.reset_buf.nonzero(as_tuple=False).reshape((-1,))
        for i in reset_env:
            self.reset_idx(i)

        # 计算reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # compute observations
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos / 3.0, -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel / 3.0, -1, 1),
                torch.clip(self.base_ang_vel / 3.14159, -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ------------ reward functions----------------
    def _reward_target(self):
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(
            torch.square(self.rel_pos), dim=1
        )
        return target_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew


if __name__ == "__main__":
    gs.init(gs.gpu)
    env = QuadEnv_polyv3(num_envs=10, show_viewer=False)
    env.reset()
    print(env.target_pos_buf.shape)
    print(env.target_pos.shape)
    print(env.rel_pos.shape)
    print(env.rel_pos)
