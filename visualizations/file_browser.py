'''
数据文件浏览器和快速可视化工具
帮助用户浏览和选择 Lab 数据和 Aw 数据文件进行比较
所有可视化功能通过调用 vis_compare 模块实现
'''

import os
import sys
import joblib
import pandas as pd
import glob
import re
from pathlib import Path
from loguru import logger


# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True, 
          format="<level>{level}</level> | {message}")

def list_lab_files(directory):
    """列出 Lab 数据目录中的所有 .joblib 文件"""
    logger.info(f"Searching for Lab data files in: {directory}")
    
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return []
    
    # 查找所有 .joblib 文件
    joblib_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if re.search("^(L|ID)[0-9]+.*.ontrackclassifier.joblib$", file) and "Walk" not in file:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, directory)
                joblib_files.append((relative_path, full_path))
    
    return sorted(joblib_files)

def list_aw_files(directory):
    """列出 Aw 数据目录中的所有 .csv 文件"""
    logger.info(f"Searching for Aw data files in: {directory}")
    
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return []
    
    # 查找所有 .csv 文件
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):  # 过滤出数据文件
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, directory)
                csv_files.append((relative_path, full_path))
    
    return sorted(csv_files)

def preview_lab_file(file_path):
    """预览 Lab 文件内容"""
    try:
        data = joblib.load(file_path)
        logger.info(f"Lab file structure:")
        
        if isinstance(data, dict):
            logger.info(f"  Type: Dictionary with keys: {list(data.keys())}")
            
            # 尝试获取数据大小
            for key, value in data.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}: Dictionary with keys: {list(value.keys())}")
                    if 'accelX' in value:
                        logger.info(f"    accelX length: {len(value['accelX'])}")
                elif isinstance(value, (list, pd.DataFrame)):
                    logger.info(f"  {key}: {type(value).__name__} with length: {len(value)}")
                else:
                    logger.info(f"  {key}: {type(value).__name__}")
        else:
            logger.info(f"  Type: {type(data).__name__}")
            if hasattr(data, '__len__'):
                logger.info(f"  Length: {len(data)}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        return False

def preview_aw_file(file_path):
    """预览 Aw 文件内容"""
    try:
        data = pd.read_csv(file_path, nrows=5)  # 只读取前5行
        logger.info(f"Aw file structure:")
        logger.info(f"  Columns: {list(data.columns)}")
        logger.info(f"  First few rows:")
        for i, row in data.iterrows():
            logger.info(f"    Row {i}: {row.tolist()}")
        
        # 获取文件总行数
        with open(file_path, 'r') as f:
            total_lines = sum(1 for _ in f) - 1  # 减去头部行
        logger.info(f"  Total data rows: {total_lines}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        return False

def select_file(files, file_type):
    """交互式文件选择"""
    if not files:
        logger.error(f"No {file_type} files found!")
        return None
    
    logger.info(f"\n=== Available {file_type} Files ===")
    for i, (relative_path, full_path) in enumerate(files):
        logger.info(f"{i+1:2d}. {relative_path}")
    
    while True:
        try:
            choice = input(f"\nSelect a {file_type} file (1-{len(files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                selected_file = files[idx][1]  # 返回完整路径
                logger.info(f"Selected: {files[idx][0]}")
                
                # 预览文件
                preview = input("Preview this file? (y/n): ").strip().lower()
                if preview == 'y':
                    if file_type == "Lab":
                        preview_lab_file(selected_file)
                    else:
                        preview_aw_file(selected_file)
                
                confirm = input("Use this file? (y/n): ").strip().lower()
                if confirm == 'y':
                    return selected_file
            else:
                logger.warning("Invalid selection. Please try again.")
                
        except ValueError:
            logger.warning("Please enter a valid number or 'q' to quit.")
        except KeyboardInterrupt:
            logger.info("\nOperation cancelled.")
            return None

def run_visualization(lab_file, aw_file):
    """
    运行可视化流程
    所有可视化逻辑都委托给 vis_compare 模块
    """
    logger.info("\n=== Starting Visualization Pipeline ===")
    logger.info(f"Lab file: {lab_file}")
    logger.info(f"Aw file: {aw_file}")
    
    try:
        # 导入 vis_compare 模块
        from vis_compare import main as vis_main
        import sys
        
        # 构造命令行参数来调用 vis_compare 的 main 函数
        original_argv = sys.argv.copy()
        sys.argv = [
            'vis_compare.py',
            '--lab_file', lab_file,
            '--aw_file', aw_file
        ]
        
        # 调用 vis_compare 的主函数
        logger.info("Delegating visualization to vis_compare module...")
        vis_main()
        
        # 恢复原始命令行参数
        sys.argv = original_argv
        
        logger.success("Visualization pipeline completed successfully!")
        
    except ImportError as e:
        logger.error(f"Could not import visualization module: {str(e)}")
        logger.info("Please ensure vis_compare.py is in the same directory")
        
        # 提供手动运行命令
        logger.info("Manual command to run visualization:")
        logger.info(f"python vis_compare.py --lab_file '{lab_file}' --aw_file '{aw_file}'")
        
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")
        
        # 提供手动运行命令作为备选
        logger.info("Manual command to run visualization:")
        logger.info(f"python vis_compare.py --lab_file '{lab_file}' --aw_file '{aw_file}'")

def main():
    logger.info("=== Lab vs Aw Data File Browser ===")

    # 列出可用文件
    lab_files = list_lab_files(lab_data_dir)
    aw_files = list_aw_files(aw_data_dir)

    if not lab_files:
        logger.error("No Lab data files found!")
        return

    if not aw_files:
        logger.error("No Aw data files found!")
        return
    
    # 选择 Lab 文件
    logger.info(f"\nFound {len(lab_files)} Lab data files")
    lab_file = select_file(lab_files, "Lab")
    if not lab_file:
        logger.info("No Lab file selected. Exiting.")
        return

    # 选择 Aw 文件
    logger.info(f"\nFound {len(aw_files)} Aw data files")
    aw_file = select_file(aw_files, "Aw")
    if not aw_file:
        logger.info("No Aw file selected. Exiting.")
        return
    
    # 确认并运行可视化
    logger.info("\n=== Ready to visualize ===")
    proceed = input("\nProceed with visualization? (y/n): ").strip().lower()
    
    if proceed == 'y':
        run_visualization(lab_file, aw_file)
    else:
        logger.info("Visualization cancelled.")
        logger.info("Manual command to run later:")
        logger.info(f"python vis_compare.py --lab_file '{lab_file}' --aw_file '{aw_file}'")

if __name__ == "__main__":
    rootDir = '/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project'
    lab_data_dir = os.path.join(rootDir, r"Data/OnTrack")
    aw_data_dir = os.path.join(rootDir, r"Data/my_data")
    main()
if __name__ == "__main__":
    rootDir = '/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project'
    lab_data_dir = os.path.join(rootDir, r"Data/OnTrack")
    aw_data_dir = os.path.join(rootDir, r"Data/my_data")
    main()

