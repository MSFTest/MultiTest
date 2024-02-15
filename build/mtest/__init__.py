import os
project_dir = os.getenv('PROJECT_DIR')
if project_dir is None:
    raise ValueError("Please set the environment variable 【PROJECT_DIR】 to the 【project path】for example, os.environ['PROJECT_DIR'] =YOUR/PATH ")
import sys
sys.path.append(project_dir)