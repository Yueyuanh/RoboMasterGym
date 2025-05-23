import open3d as o3d
import os
import sys

def simplify_mesh(input_path, output_path, ratio):
    try:
        mesh = o3d.io.read_triangle_mesh(input_path)
        if not mesh.has_triangles():
            print(f"空网格跳过: {input_path}")
            return False
        target_faces = int(len(mesh.triangles) * ratio)
        simplified = mesh.simplify_quadric_decimation(target_faces)
        simplified.compute_vertex_normals()
        o3d.io.write_triangle_mesh(output_path, simplified)
        return True
    except Exception as e:
        print(f"✘ 失败: {input_path} → {e}")
        return False

def batch_simplify(input_dir, output_dir, ratio):
    print("使用 Open3D 批量简化 STL")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"压缩率: {ratio*100:.1f}%")
    print("------------------------------------------------")

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".stl"):
                in_path = os.path.join(root, file)
                rel_path = os.path.relpath(in_path, input_dir)
                out_path = os.path.join(output_dir, rel_path)

                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                success = simplify_mesh(in_path, out_path, ratio)
                if success:
                    print(f"✔ 简化成功: {rel_path}")
                else:
                    print(f"✘ 简化失败: {rel_path}")

    print("所有文件处理完成")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python simplify_stl.py <输入目录> <输出目录> <压缩率(0.0~1.0)>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    ratio = float(sys.argv[3])
    if not (0.0 < ratio <= 1.0):
        print("压缩率必须是 0.0 到 1.0 之间的小数（不包含0）")
        sys.exit(1)

    batch_simplify(input_dir, output_dir, ratio)
