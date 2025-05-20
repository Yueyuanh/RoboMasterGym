import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch
from math import sqrt
from tracikpy import TracIKSolver
import time  
import torch
import math
import random
import cv2
import os
from scipy.spatial.transform import Rotation as R

# 初始化 Gym
gym = gymapi.acquire_gym()

# 解析命令行参数
args = gymutil.parse_arguments(
    description="Projectiles Example: Press SPACE to fire a projectile. Press R to reset the simulation.",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 4, "help": "要创建的环境数量"},
        {"name": "--capture_video", "action": "store_true", "help": "录制视频保存到本地"},
        {"name": "--headless", "action": "store_true",  "help": "无渲染模式"}
        ])

# 配置仿真参数
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.05
    sim_params.flex.num_inner_iterations = 6
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 8
    sim_params.physx.use_gpu = True

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("警告：强制使用 CPU 管线")

# 创建仿真
graphics_device_id=0
if args.headless:
    graphics_device_id=-1

sim = gym.create_sim(args.compute_device_id, graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** 创建仿真失败")
    quit()

# 添加地面
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# 创建 viewer（必须）
if args.headless:
    viewer=None
else:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** 创建 Viewer 失败")
        quit()


#<-----------------------设置初始相机视角--------------------------->
if not args.headless:

    # 从 Z 正方向观察原点
    cam_pos = gymapi.Vec3(3.0, 1.2, 0.5)  # 相机位置
    cam_target = gymapi.Vec3(0.0, 1.0, 0.0)  # 观察点（原点）
    cam_transform = gymapi.Transform()
    cam_transform.p = cam_pos
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # 监听按键/鼠标事件
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "space_shoot")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "init")
    gym.subscribe_viewer_mouse_event(viewer, gymapi.MOUSE_LEFT_BUTTON, "mouse_shoot")


#<-----------------------兑换站模型--------------------------->
asset_root = "../../assets"
asset_file = "urdf/Exchange_Station/urdf/Exchange_Station.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)



ik_solver = TracIKSolver(
    "../../assets/urdf/Exchange_Station/urdf/Exchange_Station.urdf",
    "base_link",
    "L7",
    solve_type="Distance"  # 或 "Spesed"
    # solve_type="Speed"  # 或 "Speed"
)
# 打印信息
print(f"Number of joints: {ik_solver.number_of_joints}")
print(f"Joint names: {ik_solver.joint_names}")




#<-----------------------机器人--------------------------->
robot_file = gold_file = "urdf/Gold/urdf/Gold.urdf"

robot_options = gymapi.AssetOptions()
robot_options.fix_base_link = True
robot = gym.load_asset(sim, asset_root, robot_file, robot_options)


#<-----------------------矿石--------------------------->
gold_file = "urdf/Gold/urdf/Gold.urdf"
gold_options = gymapi.AssetOptions()
gold = gym.load_asset(sim, asset_root, gold_file, gold_options)


#<-----------------------内部相机参数--------------------------->
# camera_width=512
# camera_height=256

camera_width=1920
camera_height=1080

camera_props = gymapi.CameraProperties()
camera_props.horizontal_fov = 75.0
camera_props.width = camera_width
camera_props.height = camera_height
camera_handles=[]

# 视频监视
video_fps=25
video=[]
video_writer=[]
output_path="autoaim_capture"

if args.capture_video and not os.path.exists(output_path):
    os.mkdir(output_path)

if not os.path.exists("multiple_camera_images"):
    os.mkdir("multiple_camera_images")


# 创建多个环境
num_envs = args.num_envs
num_per_row = int(sqrt(num_envs))
spacing = 2

envs = []
bullet_envs = []
actor_handles = []
robot_handles = []
gold_handles=[]

Z_Eular=-1.5707963705062866


# 初始化奖励
hit_maker=[] #击中可视化
rewards=[]
env_rewards=[]
hit_board_id=[]


station_lower=[]
station_upper=[]

#<-----------------------初始化--------------------------->
for i in range(num_envs):
    lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, lower, upper, num_per_row)
    envs.append(env)

    #<-----------------------兑换站--------------------------->
    # 设置初始姿态
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(-1, 0.1, 0.0)
    # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    # 创建四元数对象
    q = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    # 转换为欧拉角（ZYX顺序）
    yaw, pitch, roll = q.to_euler_zyx()

    # 输出结果（以度为单位）
    # print(yaw,pitch,roll)
    pose.r = gymapi.Quat.from_euler_zyx(yaw, pitch, roll) # yaw pit rol

    ahandle = gym.create_actor(env, asset, pose, "station", i, 1)
    actor_handles.append(ahandle)

    # 让机器人关节不受驱动
    props = gym.get_actor_dof_properties(env, ahandle)
    # 设置驱动模式
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(1500.0)
    props["damping"].fill(200.0)
    gym.set_actor_dof_properties(env, ahandle, props)

    station_lower=props["lower"]
    station_upper=props["upper"]

    # print(station_upper)
    # armor_body_names = gym.get_actor_rigid_body_names(env, ahandle)
    # print(armor_body_names)


    #<-----------------------机器人--------------------------->
    # 设置初始姿态
    # robot_pose = gymapi.Transform()
    # robot_pose.p = gymapi.Vec3(1, 0.0, 0.0)
    # robot_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    # robot_ahandle = gym.create_actor(env, robot, robot_pose, "robot", i, 1)
    # robot_handles.append(robot_ahandle)

    # # 设置机器人关节驱动
    # robot_props = gym.get_actor_dof_properties(env, robot_ahandle)
    # # 设置机器人PD控制器
    # # 设置驱动模式为位置控制
    # robot_props["driveMode"].fill(gymapi.DOF_MODE_POS)

    # # 设置刚度和阻尼
    # robot_props["stiffness"].fill(1000.0)  
    # robot_props["damping"].fill(100.0)     
    # robot_props['lower'].fill(-1000)
    # robot_props['upper'].fill(1000)
    # gym.set_actor_dof_properties(env, robot_ahandle, robot_props)

    # # 初始化状态
    # robot_init_states = gym.get_actor_dof_states(env, robot_ahandle, gymapi.STATE_ALL)
    # robot_init_states['pos'][:] = 0.0  # 对应第一个 DOF
    # robot_init_states['vel'][:] = 0.0
    # gym.set_actor_dof_states(env, robot_ahandle, robot_init_states, gymapi.STATE_ALL)

    # body_names = gym.get_actor_rigid_body_names(env, robot_ahandle)
    # print(body_names)

    #<-----------------------矿石--------------------------->

    # 设置初始姿态
    gold_pose = gymapi.Transform()
    gold_pose.p = gymapi.Vec3(1, 1, 0.0)
    gold_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    gold_ahandle = gym.create_actor(env, gold, gold_pose, "gold", i, 0)
    gold_handles.append(gold_ahandle)

 
    #<-----------------------内部相机--------------------------->
    camera_pos = gymapi.Vec3(0, 0, 0)
    camera_ahandle=gym.create_camera_sensor(env,camera_props)
    gym.set_camera_transform(camera_ahandle, env, gymapi.Transform(p=camera_pos, r=gymapi.Quat()))
    camera_handles.append(camera_ahandle)

    #<-----------------------结算--------------------------->




    #<-----------------------视频录制--------------------------->
    # 创建 OpenCV 视频写入器
    video_filename = os.path.join(output_path, "camera_env%d_video.mp4" % i)
    video_ahandle = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID' 保存为 .avi
    video_writer_ahandle = cv2.VideoWriter(video_filename, video_ahandle, video_fps, (camera_width, camera_height))# 相机视角
    video.append(video_ahandle)
    video_writer.append(video_writer_ahandle)



target_positions=[0,0]
target_velocity=[0,0]

aim_angle=[0,0]

tx=[]
tid=[]
vid=[]


# 使用关节选转
def station_fk(ik_solver_handle,current_joints):
        #求解器句柄,各个关节角度
        
        q = np.array(current_joints) 
        # 正解：旋转平移矩阵
        ee_pose = ik_solver_handle.fk(q)
        # 提取旋转矩阵和平移向量
        rot_mat = ee_pose[:3, :3]
        pos = ee_pose[:3, 3]
        # 转为欧拉角（单位：弧度）
        euler = R.from_matrix(rot_mat).as_euler('zyx')
        # 转为四元数
        quat = R.from_matrix(rot_mat).as_quat()  # [x, y, z, w]

        return pos,euler
        # print("位置:", pos)
        # print("欧拉角:", euler)
        # print("四元数:", quat)
    

def station_ik(ik_solver_handle,init_joints,target_positions,target_angles):
    # 目标末端位姿
    # position = [1, 1.0, 1.5]       # 位置 (x, y, z)
    # euler = [0.0, 0.0, -np.pi/2]      # 欧拉角 (roll, pitch, yaw)，单位：弧度

    # 将欧拉角转为旋转矩阵
    rot_mat = R.from_euler('zyx', target_angles).as_matrix()  # 3x3 旋转矩阵

    # 构造 4x4 位姿矩阵
    ee_pose = np.eye(4)
    ee_pose[:3, :3] = rot_mat         # 旋转部分
    ee_pose[:3, 3] = target_positions         # 平移部分

    # 初始关节角
    # qinit = np.zeros(ik_solver.number_of_joints)

    qinit = np.random.uniform(station_lower, station_upper)
    target_joints = ik_solver_handle.ik(ee_pose, qinit=init_joints)

    if target_joints is not None:
        target_joints = target_joints.astype(np.float32)  # 强制转换为 float32
        # print(target_joints)
        return target_joints,True
    
    else:
        print("IK 解失败，未找到解")
        return [0,0,0,0,0,0,0],False




# 原始位置
station_init_position=[0.8710,0.0,1.15]
station_init_angle=[-1.57,0,1.57]

# 设定原位置偏转角度
station_target_pos_bias=[0,0,0]
station_target_alg_bias=[0,0,0]
# 设定位置角度
station_target_pos=[0,0,0]
station_target_alg=[0,0,0]

# 发送目标解算位置
station_send_joints=[0,0,0,0,0,0,0]

station_send_joints_envs=[]
station_target_pos_envs=[]
station_target_alg_envs=[]

station_target_pos_bias_envs=[]
station_target_alg_bias_envs=[]


for _ in range(num_envs):
    station_send_joints_envs.append(station_send_joints)

    station_target_pos_envs.append(station_target_pos)
    station_target_alg_envs.append(station_target_alg)
    station_target_pos_bias_envs.append(station_target_pos_bias)
    station_target_alg_bias_envs.append(station_target_alg_bias)


pos_ranges = np.array([
    [ 0.00, 0.25], # x
    [-0.2, 0.2], # y
    [ 0.72-1.15, 0.90-1.15], # z
])

alg_ranges = np.array([
    [0, 1], # p
    [-1,1], # y
    [-0.5, 0.5], # r
])
   
while not gym.query_viewer_has_closed(viewer):

    gym.render_all_camera_sensors(sim)
    gym.simulate(sim)
    gym.fetch_results(sim, True)



    # <-------操作发射----------> 
    reset = False
    init  = False
    for evt in gym.query_viewer_action_events(viewer):
        if (evt.action == "space_shoot" or evt.action == "mouse_shoot") and evt.value > 0:
            reset = True
        if evt.action == "init" and evt.value > 0:
            init=True



    for i in range(num_envs):




        # 切换位置
        if reset:
            #target_position = [np.random.uniform(low, high) for low, high in ranges]

            # station_target_pos_bias=[0.0,0,0] #x 0.35,y,z
            # station_target_alg_bias=[0.0,0.0,0.5]   #p,y,r
 
            # 随机xyz
            while True:
                station_target_pos_bias_envs[i]=[np.random.uniform(low, high) for low, high in pos_ranges]
                station_target_alg_bias_envs[i]=[np.random.uniform(low, high) for low, high in alg_ranges]

                station_target_pos_envs[i]=np.array(station_init_position)+np.array(station_target_pos_bias_envs[i])
                station_target_alg_envs[i]=np.array(station_init_angle)+np.array(station_target_alg_bias_envs[i])

                # 获取解算角度
                station_send_joints_envs[i],state=station_ik(ik_solver,station_send_joints_envs[i],station_target_pos_envs[i],station_target_alg_envs[i])

                if state:
                    break

        if init:
            station_send_joints_envs[i].fill(0)

        gym.set_actor_dof_position_targets(envs[i], actor_handles[i], station_send_joints_envs[i])

    # 渲染更新
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# print(rewards)
# print(env_rewards)

# 释放资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


#

