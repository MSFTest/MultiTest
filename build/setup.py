from setuptools import setup,find_packages

if __name__ == '__main__':

      setup(name='mtest', #包名,通常写package_name
            install_requires=['numpy',
                              "open3d",
                              'tqdm',
                              "astropy",
                              ], # 依赖要求
            python_requires=">=3.7", #python版本要求
            version='1.0', #版本号
            description='MultiTest: Physical-Aware Object Insertion for Testing Multi-sensor Fusion Perception Systems',	#描述
            author='niangao',	#作者
            author_email='xinyugao@smail.nju.edu.cn',	#邮箱
            include_package_data=True,	#whl包是否包含除了py文件以外的文件
            packages=find_packages()
      )

