# Image_Caption_and_Matching
A project for ML 2020 class. A mission based on Flickr30k images, which is aimed at matching sentences in a pool to each image from the image set.
This is the final Course Project for ML2020, which is aimed at generating captions when given a picture, and find the best 5 fitted other ones in the Pool.

The code is mainly based on https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning, yet the Attention model is changed into an Adaptive Attention model.
The reference for the modification is 'Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning'（https://arxiv.org/abs/1612.01887，2017)
Based on the course dataset & pool, the performance after the modification is a little bit better.

1. 文件:
(1）PreProcess.py：数据预处理。输入为Train_Caption.txt划分出来的训练集和测试集txt文件，输出为json格式文件；

(2）create_input_files.py：数据预处理，来自开源代码。在训练之前需运行，需修改json格式文件路径；

(3) utils.py：工具文件，来自开源代码。不能直接运行；

(4) train.py：用于训练，使用适应性注意力机制模型，使用预训练词嵌入层，修改自开源代码文件。修改数据路径后可以直接运行；

(5) eval.py：用于验证，使用适应性注意力机制模型，修改自开源代码文件。修改数据、模型路径后可以直接运行；

(6) Adamodels.py：适应性注意力机制模型。不能直接运行；

(7) caption.py：用于注意力及描述语句可视化，使用适应性注意力机制模型，修改自开源代码。根据提示输入，输出为测试图片的注意力及描述语句可视化结果。命令行示例：python caption.py --model='./BEST_checkpoint_flickr30k_5_cap_per_img_5_min_word_freq.pth.tar' --img='/dataset/Flickr30k_Images/533044058.jpg' --word_map='../dataset/modified/processed_data/WORDMAP_flickr30k_5_cap_per_img_5_min_word_freq.json' --beam_size=3

(8) matching.py：用于语句匹配，使用适应性注意力机制模型，根据提示输入，输出为result.txt，其中包含匹配结果。命令行示例：python matching.py --model='./BEST_checkpoint_flickr30k_5_cap_per_img_5_min_word_freq.pth.tar' --modeltype='AdaAttention' --pool='../dataset/Test_CaptionPool.txt' --word_map='../dataset/modified/processed_data_
WORDMAP_flickr30k_5_cap_per_img_5_min_word_freq.json' --beam_size=3

(9) datasets.py：数据集文件，来自开源代码。不能直接运行；

(10) my_cider.py：CIDEr指标，修改自开源代码。由train.py，eval.py，matching.py引用；

(11) get_pretrained_embeddings.py：GloVe与训练词嵌入层。由train.py引用；

2. 环境:
python 3.7.9
Pytorch 1.7.1
scipy 1.1.0
pillow 8.0.1
cudatoolkit 10.1.243
h5py 2.10.0
imageio 2.9.0
matplotlib  3.3.2
numpy 1.19.2

3. It's my first project here, see it as a test. Thank you for reading.

