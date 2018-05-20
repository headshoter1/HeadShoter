# HeadShoter
Deep nueral network playing GTA V
Project based on TensorFlow API to locate head of game character and shot it from the first time. It use faster_rcnn_inception_v2_coco_2018_01_28 to detecting person in scene of the game. Than HeadShoter finds position of head to shoot. As base for project I was used project of Sentdex https://github.com/sentdex/pygta5, but this guy using it for self-driving car. Now network is working good only if there isn't many characters in the scene and position of head is approximate. But I with my girl are working for teaching network detecting head like separate object to find exact position of head in the future. And It'll go from character to character one by one.
Let me express my deep gratitude for Sentdex https://github.com/Sentdex for his amazing tutorials https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ
And also my gratitude and best regards for Evgeniy Vesnin (technical director of Mallenom Systems company) for introducing me into the subject of deep neural network  

So, what do you need to run this projec?

1. Download and install https://www.tensorflow.org/install/
2. Have GPU with CUDA 9 and install CUDA Toolkit https://developer.nvidia.com/cuda-90-download-archive
3. Download and install https://github.com/tensorflow/models/tree/master/research/object_detection
