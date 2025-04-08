# MultiTest

This repository provides the code of the paper "**MultiTest: Physical-Aware Object Insertion for Testing Multi-sensor Fusion Perception Systems**"

[[website]](https://sites.google.com/view/msftest)

![](https://github.com/853108389/MultiTest/blob/master/src/workflow.png)
MultiTest employs a physical-aware approach to render modality-consistent object instances using virtual sensors to for Testing Multi-sensor Fusion (MSF) Perception Systems. 

Figure above presents the high-level workflow of MultiTest.  Given a background multi-modal data recorded from real-world and an object instance selected from the object database, MultiTest first executes the pose estimation module to calculate the valid locations and orientations of an object to be inserted. Then the multi-sensor simulation module renders the object instance in the form of both image and point cloud given the calculated poses in a physical-aware virtual simulator. The multi-sensor simulation module further merges the synthesized image and point cloud of the inserted object with the background data and carefully handles the occlusion. These two modules form the MultiTest’s multi-modal test data generation pipeline. Finally, the realistic multi-modal test data can be efficiently generated through fitness guided metamorphic testing. We detail each module of MultiTest in the following.



## The structure of the repository 

Main Folder Structure:

```
The main folder structure is as follows:
MultiTest
├── _assets     
│   └── shapenet                         object database                 
├── _datasets							
│   ├── kitti                            kitti dataset
│   └── kitti_construct                  generate test cases
├── _queue_guided                        seed queue
├── system                               systems under test
├── blender                              blender script
├── config                               sensor and algorithm configuration
├── build                                package building for MultiTest
├── third                                third-party repository
├── eval_tools                           tools for evaluation AP
├── build_script.py                      package building script
├── evaluate_script.py                   system evaluating script
├── fitness_score.py                     fitness metric calculation
├── init.py                              environment setup script
├── logger.py                            log                  
├── visual.py                            data visualisation script
├── demo.py                              quick start demo
└── main.py                              MultiTest main file
```



## Installation

We implement all the MSF systems with PyTorch 1.8.0 and Python 3.7.11 in the Linux environment. All experiments are conducted on a server with an Intel i7-10700K CPU (3.80 GHz), 48 GB RAM, and an NVIDIA GeForce RTX 3070 GPU (8 GB VRAM). 

### Basic Dependency

Run the following command to install the dependencies

```bash
pip install -r requirements.txt
python build_script.py
```

Set your project path `config.common_config.project_dir="YOUR/PROJECT/PATH"`

### Quick Start

1. Install blender.

   MultiTest leverage blender, an open-source 3D computer graphics software, to build virtual
   camera sensor. 

   - install blender>=3.3.1 from this [link](https://www.blender.org/download/)
   - setting the config `config.camera_config.blender_path="YOUR/BLENDER/PATH"`

2. Install S2CRNet **[optional]**.

   MultiTest leverage S2CRNet to improve the realism of the synthesized test cases.

   - download repo from [link](https://github.com/stefanLeong/S2CRNet) to `MultiTest/third/S2CRNet` 

     `git clone git@github.com:stefanLeong/S2CRNet.git`

   - setting the config `config.camera_config.is_image_refine=True`

3. Install CENet **[optional]**.
   MultiTest leverage CENet to split road from point cloud and get accurate object positions.

   - download repo from [link](https://github.com/huixiancheng/CENet) to `MultiTest/third/CENet` 
     `git clone git@github.com:huixiancheng/CENet.git`

After installing all the necessary configurations, you can  run the `demo.py` file we provided to generate multi-modal data:

```bash
python init.py
python demo.py 
```

The result can be found at `MultiTest/_datasets/kitti_construct/demo`. Then we can run `visual.py` to visualize the synthetic data



## Complete Requirements

In order to reproduce our experiment, we should install the complete dependency.  Before that, we should install all the dependencies from the "Quick Start" section.

### Install MSF Perception Systems 

In order to reproduce our experiments, we need to carefully configure the environment for each system.

These system are derived from the MSF benchmark. Detailed configuration process are provided [here](https://sites.google.com/view/ai-msf-benchmark/replication-package).

These systems  should be placed in the directory `MultiTest/system/SYSTEM_NAME`

### Download Datasets 

1. KITTI
   - Download KITTI datasets from this [link](https://www.cvlibs.net/datasets/kitti/index.php) to `MultiTest/_datasets/kitti`
2. ShapeNet
   - Download ShapeNet datasets from this [link](https://shapenet.org/) to `MultiTest/_assets/shapnet`
   - Refer to this [link](https://github.com/CesiumGS/obj2gltf) to create a 3D model in gltf format.

### Generate Multi-modal data with Guidance

```bash
python main.py --system_name "SYSTEM" --select_size "SIZE" --modality "multi"
```

The result can be found at `MultiTest/_datasets/kitti_construct/SYSTEM`.



## Experiments 

##### RQ1: Realism Validation

1. Generation of multimodal data from 200 randomly selected seeds.

   ```bash
   python main.py --system_name random --select_size 200
   ```

2. Validating the realism of synthetic image.

   - Install pytorch-fid from [here](https://github.com/mseitzer/pytorch-fid)

     ```bash
     pip install pytorch-fid
     ```

   - Usage

     ```bash
     python -m pytorch_fid "MultiTest/datasets/kitti/training/image_2" "/MultiTest/_datasets/kitti_construct/SYSTEM/training/image_2"
     ```

3. Validating the realism of synthetic LiDAR point cloud.

   - Install frd from [here](https://github.com/vzyrianov/lidargen)

   - Usage

     ```bash
     python lidargen.py --fid --exp kitti_pretrained --config kitti.yml
     ```

4. Validating the modality-consistency of synthetic multi-modal data.

   ​	The result can be found at `Multimodality/RQ/RQ1/consistent`.

##### RQ2: Fault Detection Capability

1. Generation of multimodal data with fitness guidance from 200 randomly selected seeds.

   ```bash
   python main.py --system_name "SYSTEM" --select_size 200
   ```

2. Evaluate the AP value and the number of errors with each error category on the generated test cases of a perception system.

   ```bash
   python RQ2_tools.py --system_name "SYSTEM" --seed_num 200 iter=1
   
   ```

##### RQ3: Performance Improvement

1. Formatting the generated data into KITTI format of a perception system for retraining

   ```bash
   python copy_data.py --system_name "SYSTEM"
   
   ```

   The retraining dataset can be found at `_workplace_re/SYSTEM/kitti`.

   2. Copy the dataset to the dataset directory of the appropriate system and execute the training script provided by each systems.

      

## Custom Configuration

Run MultiTest on a custom dataset： 

1. Prepare dataset in KITTI dataset format.
2. Set your dataset path `config.common_config.kitti_dataset_root="YOUR/DATASET/PATH" `

Run MultiTest with custom 3D models：

1. Prepare your model files in gltf format. 
2. Set your model path `config.common_config.assets_dir ="YOUR/ASSETS/PATH"`

Run MultiTest with custom MSF systems：

1. Place the system in the directory  `MultiTest/system/YOUR_SYSTEM_NAME`
2. Provide the inference interface at line 548 of the `main.py` file
