
import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch
from math import sqrt
import time  
import torch
import math
import random

# 初始化 Gym
gym = gymapi.acquire_gym()

# 解析命令行参数
args = gymutil.parse_arguments(
    description="Projectiles Example: Press SPACE to fire a projectile. Press R to reset the simulation.",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 4, "help": "要创建的环境数量"}])

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
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** 创建仿真失败")
    quit()

# 添加地面
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# 创建 viewer（必须）
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** 创建 Viewer 失败")
    quit()

# 设置初始相机视角：从 Z 正方向观察原点
cam_pos = gymapi.Vec3(1.0, 1.0, 3.0)  # 相机位置
cam_target = gymapi.Vec3(0.0, 0.0, 0.0)  # 观察点（原点）
cam_transform = gymapi.Transform()
cam_transform.p = cam_pos
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 监听按键/鼠标事件
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "space_shoot")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
gym.subscribe_viewer_mouse_event(viewer, gymapi.MOUSE_LEFT_BUTTON, "mouse_shoot")

#<-----------------------靶装甲模型--------------------------->
asset_root = "../../assets"
asset_file = "urdf/Armor_Robot/urdf/Armor_Robot.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

armor_link_1_index = gym.find_asset_rigid_body_index(asset, "Armor_board_link_1")
armor_link_2_index = gym.find_asset_rigid_body_index(asset, "Armor_board_link_2")
armor_link_3_index = gym.find_asset_rigid_body_index(asset, "Armor_board_link_3")
armor_link_4_index = gym.find_asset_rigid_body_index(asset, "Armor_board_link_4")


#<-----------------------机器人--------------------------->
robot_file = "urdf/Wheelleg_Robot/urdf/Wheelleg_Robot.urdf"
# robot_file ="urdf/Armor_Robot/urdf/Armor_Robot.urdf"
robot_options = gymapi.AssetOptions()
robot_options.fix_base_link = True
robot = gym.load_asset(sim, asset_root, robot_file, robot_options)

#<-----------------------弹丸--------------------------->
bullet_asset_options = gymapi.AssetOptions()
bullet_asset_options.density = 10.0

#  创建球形发射体（半径为 0.017）
bullet_asset = gym.create_sphere(sim, 0.017, bullet_asset_options)
bullet_handles = []
bullet_index=[]

#<-----------------------LED--------------------------->
led_asset_options = gymapi.AssetOptions()
led_asset_options.density = 10.0
led_asset = gym.create_sphere(sim, 0.05, led_asset_options)
led_handles = []



# 创建多个环境
num_envs = args.num_envs
num_per_row = int(sqrt(num_envs))
spacing = 2

envs = []
bullet_envs = []
actor_handles = []
robot_handles = []
Z_Eular=-1.5707963705062866


# 初始化奖励
hit_maker=[] #击中可视化
rewards=[]
env_rewards=[]


for i in range(num_envs):
    lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, lower, upper, num_per_row)
    envs.append(env)

    #<-----------------------靶装甲--------------------------->
    # 设置初始姿态
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(-1, 0.0, 0.0)
    # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    # 创建四元数对象
    q = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    # 转换为欧拉角（ZYX顺序）
    yaw, pitch, roll = q.to_euler_zyx()

    # 输出结果（以度为单位）
    # print(yaw,pitch,roll)
    pose.r = gymapi.Quat.from_euler_zyx(yaw, pitch+1.57, roll) # yaw pit rol

    ahandle = gym.create_actor(env, asset, pose, "armor", i, 1)
    actor_handles.append(ahandle)

    # 让机器人关节不受驱动
    props = gym.get_actor_dof_properties(env, ahandle)
    # 设置驱动模式
    props["driveMode"][0] = gymapi.DOF_MODE_POS      # 第一个关节为位置控制
    props["driveMode"][1] = gymapi.DOF_MODE_VEL      # 第二个关节为速度控制

    # 设置刚度与阻尼（仅位置控制有效）
    props["stiffness"][0] = 1000.0
    props["damping"][0] = 200.0

    # 速度控制模式下，stiffness 应为 0，一般只使用 damping 控制阻尼
    props["stiffness"][1] = 1000.0
    props["damping"][1] = 50.0
    gym.set_actor_dof_properties(env, ahandle, props)

    #<-----------------------机器人--------------------------->
    # 设置初始姿态
    robot_pose = gymapi.Transform()
    robot_pose.p = gymapi.Vec3(1, 0.0, 0.0)
    robot_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    robot_ahandle = gym.create_actor(env, robot, robot_pose, "robot", i, 1)
    robot_handles.append(robot_ahandle)

    # 设置机器人关节驱动
    robot_props = gym.get_actor_dof_properties(env, robot_ahandle)
    # 设置机器人PD控制器
    # 设置驱动模式为位置控制
    robot_props["driveMode"].fill(gymapi.DOF_MODE_POS)

    # 设置刚度和阻尼
    robot_props["stiffness"].fill(1000.0)  
    robot_props["damping"].fill(100.0)     
    gym.set_actor_dof_properties(env, robot_ahandle, robot_props)

    dof_props = gym.get_actor_dof_properties(env, robot_ahandle)
    dof_props['lower'][0] = -3.14
    dof_props['upper'][0] = 3.14
    dof_props['lower'][1] = -3.14
    dof_props['upper'][1] = 3.14
    print("Lower limits:", dof_props['lower'])
    print("Upper limits:", dof_props['upper'])
    
    gym.set_actor_dof_properties(env, robot_ahandle, dof_props)

    #<-----------------------弹丸--------------------------->
    env_bullet_handles = []  # 当前环境下的所有 bullet handle
    for k in range(10):
        bullet_pose = gymapi.Transform()
        bullet_pose.p = gymapi.Vec3(0, 0, k*0.1-0.5)  
        bullet_pose.r = gymapi.Quat(0, 0, 0, 1)

        bullet_ahandle = gym.create_actor(env, bullet_asset, bullet_pose, "bullet"+ str(k), i, 0)

        # 设置颜色
        bullet_color = [0,0.8,0]
        gym.set_rigid_body_color(env, bullet_ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                gymapi.Vec3(bullet_color[0], bullet_color[1], bullet_color[2]))
        env_bullet_handles.append(bullet_ahandle)
    
    #<-----------------------指示灯--------------------------->
    led_pose = gymapi.Transform()
    led_pose.p = gymapi.Vec3(-0.5, 0, 0)  
    led_pose.r = gymapi.Quat(0, 0, 0, 1)

    led_ahandle = gym.create_actor(env, led_asset, led_pose, "led", i, 0)

    # 设置颜色
    led_color = [0.8,0,0]
    gym.set_rigid_body_color(env, led_ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                            gymapi.Vec3(led_color[0], led_color[1], led_color[2]))
    led_handles.append(led_ahandle)

    #<-----------------------结算--------------------------->

    # 弹丸序列
    bullet_index.append(0)
    # 大列表
    bullet_handles.append(env_bullet_handles)

    # 击中奖励
    rewards.append([0,0,0,0])
    env_rewards.append(0)
    hit_maker.append(0)



target_positions=[0,0]
target_velocity=[0,0]

aim_angle=[0,0]

tx=[]
tid=[]
vid=[]
# 使用关节选转
def Armor_move(handles,pos,pos_random,vel,vel_random):
    # global tx,tid

    for i in range(num_envs):
        tx.append(random.uniform(-pos_random, pos_random))
        tid.append(0)
        vid.append(random.uniform(-vel_random, vel_random))

    for i in range(num_envs):
        tid[i]+=tx[i]+pos
        target_position=math.sin(tid[i])

        target_velocity[1]=vel+vid[i]

        # 应用新的目标位置
        gym.set_actor_dof_position_targets(envs[i], handles[i], target_position)
        gym.set_actor_dof_velocity_targets(envs[i], handles[i], target_velocity)



def draw_wire_sphere(gym, viewer, env, center, radius=0.1, segments=32, color=gymapi.Vec3(1.0, 0.0, 0.0)):
    """
    用 3 个互相垂直的圆环模拟一个球体标记
    """
    def draw_circle_in_plane(normal):
        for i in range(segments):
            theta1 = 2 * math.pi * i / segments
            theta2 = 2 * math.pi * (i + 1) / segments

            # 坐标计算：根据不同法线确定圆所在平面
            if normal == 'xy':
                p1 = (center.x + radius * math.cos(theta1), center.y + radius * math.sin(theta1), center.z)
                p2 = (center.x + radius * math.cos(theta2), center.y + radius * math.sin(theta2), center.z)
            elif normal == 'yz':
                p1 = (center.x, center.y + radius * math.cos(theta1), center.z + radius * math.sin(theta1))
                p2 = (center.x, center.y + radius * math.cos(theta2), center.z + radius * math.sin(theta2))
            elif normal == 'xz':
                p1 = (center.x + radius * math.cos(theta1), center.y, center.z + radius * math.sin(theta1))
                p2 = (center.x + radius * math.cos(theta2), center.y, center.z + radius * math.sin(theta2))

            gymutil.draw_line(gymapi.Vec3(*p1), gymapi.Vec3(*p2), color, gym, viewer, env)

    # 绘制三个方向的圆圈，组成球形标记
    draw_circle_in_plane('xy')
    draw_circle_in_plane('yz')
    draw_circle_in_plane('xz')





# 主循环
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # <-------操作发射----------> 
    shoot_triggered = False
    for evt in gym.query_viewer_action_events(viewer):
        if (evt.action == "space_shoot" or evt.action == "mouse_shoot") and evt.value > 0:
            shoot_triggered = True

    # <-------目标移动----------> 
    target_pos=0.01
    target_vel=2

    # 变化范围
    target_pos_random=0.001
    target_vel_random=0.5
    
    Armor_move(actor_handles,target_pos,target_pos_random,target_vel,target_vel_random)


    # <-------机器人瞄准----------> 
    for i in range(num_envs):

        pitch_link_index = gym.find_actor_rigid_body_index(envs[i], robot_handles[i], "Pitch_link", gymapi.DOMAIN_ENV)
        # pitch_transform = gym.get_rigid_transform(envs[i], pitch_link_index)
        pitch_transform = gym.get_rigid_transform(envs[i], 17)


        # 子弹初始位置
        spawn = pitch_transform.p

        # 定义在 Pitch_link 局部坐标系中的偏移量
        local_offset = gymapi.Vec3(0.22, 0.0, 0.0)
        # 将局部偏移转换为世界坐标系中的位置
        spawn_position = pitch_transform.transform_point(local_offset)

        # 特权信息，装甲板中间位置
        armor_link_index = gym.find_actor_rigid_body_index(envs[i], actor_handles[i], "move_link", gymapi.DOMAIN_ENV)
        armor_transform = gym.get_rigid_transform(envs[i], armor_link_index)
        aim_position=armor_transform.p

        # print("Pitch link index:", pitch_link_index)
        # print(i)
        # print(aim_position,spawn)

        # 计算方向向量 
        direction = gymapi.Vec3(
            aim_position.x - spawn_position.x,
            aim_position.y - spawn_position.y,
            aim_position.z - spawn_position.z
        )
        # print(direction)
        # 归一化
        length = math.sqrt(direction.x**2 + direction.y**2 + direction.z**2)
        direction_normalized = gymapi.Vec3(
            direction.x / length,
            direction.y / length,
            direction.z / length
        )
        # print(direction_normalized)

        pitch_set = math.atan2(direction_normalized.y, direction_normalized.x)
        yaw_set = -math.atan2(-direction_normalized.z, math.sqrt(direction_normalized.x**2 + direction_normalized.y**2))
        # print(yaw_set,pitch_set)

        aim_angle[0]=yaw_set
        aim_angle[1]=pitch_set+3.14-0.05
        # aim_angle[1]-=0.05
        # print(aim_angle)

        # aim_angle[0] = np.clip(aim_angle[0], -3.14, 3.14)
        # aim_angle[1] = np.clip(aim_angle[0], -3.14, 3.14)

        # aim_angle[0]=0
        # aim_angle[1]=0.08
        gym.set_actor_dof_position_targets(envs[i], robot_handles[i], aim_angle)


        #  <-------可视化更新----------> 

        gym.clear_lines(viewer)
        draw_wire_sphere(gym, viewer, envs[i], center=spawn_position, radius=0.017)

        # 装甲坐标系
        armor_link_pos = gym.find_actor_rigid_body_index(envs[i], actor_handles[i], "Armor_board_link_1", gymapi.DOMAIN_ENV)
        armor_board_transform = gym.get_rigid_transform(envs[i], armor_link_pos)
        aim_board_position=armor_board_transform.p

        center_point=gymapi.Vec3(aim_board_position.x,aim_board_position.y+0.3,aim_board_position.z)

        led_handle=led_handles[i]
        led_state = gym.get_actor_rigid_body_states(envs[i], led_handle, gymapi.STATE_NONE)
        led_state['pose']['p'].fill((center_point.x, center_point.y, center_point.z))
        led_state['pose']['r'].fill((0, 0, 0, 1))
        gym.set_actor_rigid_body_states(envs[i], led_handle, led_state, gymapi.STATE_ALL)
    

        #  <-------射击操作----------> 

        # 之后根据每个环境的触发条件射击
        if shoot_triggered:
            # 子弹方向（假设沿着局部 Z 轴发射）
            forward_vector = gymapi.Vec3(1, 0, 0)
            bullet_direction = pitch_transform.r.rotate(forward_vector)

            # 设置子弹速度
            base_speed = 25
            noise_amplitude = 1.0  # 最大扰动范围（单位：m/s）
            bullet_speed = base_speed + random.uniform(-noise_amplitude, noise_amplitude)
            # print(bullet_speed)
            bullet_velocity = gymapi.Vec3(
                bullet_direction.x * bullet_speed,
                bullet_direction.y * bullet_speed,
                bullet_direction.z * bullet_speed
            )
            angvel = 1.57 - 3.14 * np.random.random(3)

            # 
            bullet_handle=bullet_handles[i][bullet_index[i]]
            state = gym.get_actor_rigid_body_states(envs[i], bullet_handle, gymapi.STATE_NONE)
            state['pose']['p'].fill((spawn_position.x, spawn_position.y, spawn_position.z))
            state['pose']['r'].fill((0, 0, 0, 1))
            state['vel']['linear'].fill((bullet_velocity.x, bullet_velocity.y, bullet_velocity.z))
            state['vel']['angular'].fill((angvel[0], angvel[1], angvel[2]))
            gym.set_actor_rigid_body_states(envs[i], bullet_handle, state, gymapi.STATE_ALL)

            bullet_index[i] = (bullet_index[i] + 1) 
            if bullet_index[i]>=10:
                bullet_index[i]=0
            
        # 击打检测
        # 获取接触力张量
        net_contact_force_tensor = gym.acquire_net_contact_force_tensor(sim)
        net_contact_force = gymtorch.wrap_tensor(net_contact_force_tensor)
        gym.refresh_net_contact_force_tensor(sim)

        # 获取特定环境中机器人的刚体索引
        actor_handle=actor_handles[i]
        armor_link_indices = [
            gym.find_actor_rigid_body_index(envs[i], actor_handle, "Armor_board_link_1", gymapi.DOMAIN_SIM),
            gym.find_actor_rigid_body_index(envs[i], actor_handle, "Armor_board_link_2", gymapi.DOMAIN_SIM),
            gym.find_actor_rigid_body_index(envs[i], actor_handle, "Armor_board_link_3", gymapi.DOMAIN_SIM),
            gym.find_actor_rigid_body_index(envs[i], actor_handle, "Armor_board_link_4", gymapi.DOMAIN_SIM)
        ]




        is_target=0
        # 检测每个装甲板的接触力
        for idx, body_index in enumerate(armor_link_indices):
            contact_force = net_contact_force[body_index]
            force_magnitude = torch.norm(contact_force).item()
            threshold=0.01
            if force_magnitude > threshold:
                print(f"Env_{i+1}: Armor_{idx+1} 被击中！")
                # 增加对应的奖励
                rewards[i][idx] += 1
                # 环境奖励总和
                env_rewards[i]=sum(rewards[i])
                is_target=1

        # 击中提示
        if(is_target):
            hit_maker[i]=50
            is_target=0

        if hit_maker[i]>0:
            hit_maker[i]-=1

        if hit_maker[i]>0:
            led_color_ok = [0,0.8,0]
        else:
            led_color_ok = [0.8,0,0]

        gym.set_rigid_body_color(envs[i], led_handles[i], 0, gymapi.MESH_VISUAL_AND_COLLISION,gymapi.Vec3(led_color_ok[0], led_color_ok[1], led_color_ok[2]))
        print(hit_maker)

    # 根据rewards更新
    # if sum(rewards)>0:
    #     gym.clear_lines(viewer)
    #     for idx in range(len(rewards)):
    #         for num in range(rewards[idx]):
    #             center_point=gymapi.Vec3(aim_board_position[idx].x,aim_board_position[idx].y+0.5+(0.2*num),aim_board_position[idx].z)
    #             draw_hit_count(gym, viewer, env, center_point, rewards[idx])
    #             print(center_point)

    # 渲染更新
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

print(rewards)
print(env_rewards)
# 释放资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


#

