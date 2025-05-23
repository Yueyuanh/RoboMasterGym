# RoboMasterGym

RoboMaster IsaacGym Framework

> 本项目目前只是调试完成部分项目与IsaacGym的相关接口、资产配置和规则设置，暂时并没有引入强化学习框架和相关算法（后续考虑引入LeggedGym等框架）
## TODO

- [x] 自瞄小陀螺
- [x] 工程自动兑矿
- [ ] 自动打符
- [ ] 轮腿训练
- [ ] 哨兵导航
- [ ] 战术推演


## AutoAim
| 静态截图 | 内录第一视角 |
|----------|--------------|
| <img src="doc/autoaim.png" width="350"> | <img src="doc/autoaim_1st.gif" width="350"> |
| <img src="doc/autoaim_test.gif" width="350"> | <img src="doc/autoaim_multi.gif" width="350"> |

## Exchange
| 内录第一视角 | 第三方视角 |
|----------|------------|
| <img src="doc/exchange.png" width="350"> | <img src="doc/exchange_3rd.gif" width="350"> |
| <img src="doc/exchange_base.gif" width="350"> | <img src="doc/exchange_multi.gif" width="350"> |

## 待补充

## URDF压缩

从SW导出URDF时，对于相对复杂的机器人导出STL模型面数过多，导致模型体积过大IsaacGym加载缓慢或失败，所以本项目采用Open3D对URDF meshes进行批量压缩，可大大优化模型体积与导入速度

```
python simplify_stl.py <输入目录> <输出目录> <压缩率(0.0~1.0)>
```
## 开源引用
>[RM2024-工程机器人机械结构开源上海交通大学-云汉交龙战队](https://bbs.robomaster.com/article/54080?source=4)