## GSoC 2024 Tests :mag: :white_check_mark:

  ### Project: Learning Representation Through Self-Supervised Learning on Real Gravitational Lensing Images

Deep learning has transformed the analysis of supervised lensing data, utilizing feature spaces to uncover latent variables related to dark matter. This approach, essential due to the vast amount of data, evolved from manual labeling to using unsupervised methods, like variational autoencoders, for unbiased dark matter analysis. Recently, self-supervised learning, an innovative technique that generates its own supervisory signals, has shown promise. Stein et al. applied this method using a convolutional neural network (CNN) on simulated strong gravitational lens images, outperforming traditional supervised methods. This was supported by research I contributed to in the DeepLense project during Google Summer of Code (GSoC) 2023. However, the application to real images remains underexplored. My GSoC project proposal aims to bridge this gap by applying advanced Self-Supervised Learning (SSL) techniques to real-world datasets and comparing these findings with those from simulated datasets. This will assess the reliability of simulated data for deep learning inference, marking a significant step towards leveraging real datasets for similar studies.

  

[:arrow_right: Click Here :arrow_left:](https://drive.google.com/drive/folders/1lJHjNkfqu4mm69brzNPIFmywLEF_tvma?usp=sharing) to access the data including the trained models the tests.

  
  
  

Everything is built in Keras and Tensorflow.

  

Summary:

-  `1) Common Test I. Multi-Class Classification`

-  `2) Specific Test VI. SSL on Real Dataset.`


---

### Details and results for all tasks:

  

*  **TEST I:** Multi-Label Classification 

  
| Approaches | Val AUC  | Confusion Matrix and ROC plot  |
|---|---|---|
|*`Vision Transformer (Custom)`*<br><br>Notebook: [.ipynb](https://github.com/yaashwardhan/2024-DeepLense-Tests/blob/main/Common%20Test%20I.%20Multi-Class%20Classification/Method1_ViT.ipynb)<br>This approach involves<br>a self-attention Vision Transformer<br>whoes architecture implemented from<br>scratch and then imagenet<br>pretrained weights are applied to it.<br>The model processes image patches<br>through 12 transformer blocks with<br>multi-head self-attention and MLP,<br>then outputs class probabilities. |0.90|<img src="Common Test I. Multi-Class Classification/results/Custom_ViT.png" width="600">
|*`ResNet50 Transfer Learning`*<br><br>Notebook: [.ipynb](https://github.com/yaashwardhan/2024-DeepLense-Tests/blob/main/Common%20Test%20I.%20Multi-Class%20Classification/Method2_ResNet50.ipynb)<br>Utilizing ResNet-50 for transfer<br>learning, we remove its classification<br>head, apply batch normalization,<br>dropout, and a dense<br>layer with softmax activation<br>for 3-class probability output.<br>This implementation is simplified<br>using existing libraries for<br>the model's architecture.|0.98| <img src="Common Test I. Multi-Class Classification/results/ResNet50_results.png" width="600">

---

*  **TEST VI :** Self Supervised Learning on Real Dataset

  

| Approach | Val AUC | Confusion Matrix and ROC plot |
|---|---|---|
|*`Self-Attention-CNNs`*<br><br>Notebook: [.ipynb](https://github.com/yaashwardhan/2024-DeepLense-Tests/blob/main/Specific%20Test%20VI.%20SSL%20on%20Real%20Dataset/SSL_Contrastive_Rotation_and_Gaussian.ipynb)<br>A multimodal model using CNNs<br>and attention mechanisms to process<br>images and features.<br>The model combines the image and <br>feature branches, applies self<br>attention,and outputs a probability<br>through Dense layers. | 0.99 |<img src="Task2 - Lens Finding (0.99 AUC) (Self-Attention CNN)/lens_finding_results.png" width="600">

  

---
