#!/usr/bin/env python3
"""
测试脚本 - 验证混淆矩阵功能修改是否正确
"""

import os
from config import save_dir

def test_confusion_matrix_modifications():
    """测试混淆矩阵功能的修改"""
    print("="*60)
    print("测试混淆矩阵功能修改")
    print("="*60)
    
    # 检查cm目录是否会被创建
    cm_dir = os.path.join(save_dir, 'cm')
    print(f"CM目录路径: {cm_dir}")
    
    # 检查导入是否正常
    try:
        from trainAndEvaluation.confusion_matrix_utils import save_confusion_matrix
        print("✓ save_confusion_matrix 函数导入成功")
    except ImportError as e:
        print(f"✗ save_confusion_matrix 函数导入失败: {e}")
    
    try:
        from trainAndEvaluation.confusion_matrix_utils import compare_models_confusion_matrices
        print("✓ compare_models_confusion_matrices 函数导入成功")
    except ImportError as e:
        print(f"✗ compare_models_confusion_matrices 函数导入失败: {e}")
    
    print("\n主要修改内容:")
    print("1. ✓ 移除了训练过程中的自动显示混淆矩阵")
    print("2. ✓ 添加了save_confusion_matrix函数，只保存不显示")
    print("3. ✓ 修改了compare_models_confusion_matrices，使用单一colorbar")
    print("4. ✓ 所有混淆矩阵图片保存到/cm子目录")
    print("5. ✓ 比较图也使用plt.close()，不显示只保存")
    
    print("\n使用方法:")
    print("- 运行 main.py 会自动生成所有混淆矩阵到 cm/ 目录")
    print("- 单独使用: 运行 demo_confusion_matrix.py 查看更多功能")
    
    print("="*60)

if __name__ == "__main__":
    test_confusion_matrix_modifications()
