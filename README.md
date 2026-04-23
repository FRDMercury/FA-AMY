# FA-Amy

**FA-Amy: A Deep Learning Framework for Amyloid Protein Prediction**

---

##  Introduction
**FA-Amy** is a deep learning-based framework designed to accurately predict amyloid proteins by integrating **global and local features**.  
Our model demonstrates strong robustness even under **imbalanced dataset conditions**, making it practical for real-world biological applications.

---

##  Features
- End-to-end deep learning framework
- Large learning model extracted features
- Incorporates **global + local  features**
- Robust performance on **imbalanced datasets**
- Easy-to-use training and evaluation scripts

---

##  Environment

Python Version: 3.9 or higher (Tested on 3.11)

Operating System: Linux

##  Usage


FA-Amy relies on a large-scale pre-trained protein language model: ESM C. The model is implemented using Hugging Face's Transformers library and PyTorch. Please make sure to install the required dependencies beforehand.

ESM C: https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12

First, you need to download the model weights for ESM C  from the provided Hugging Face URLs. Please visit the above links to download the respective weight files.

Save the downloaded weight files in your working directory and make sure you know their exact paths.

Next, you will use the provided `ESM_embedding.py` script to generate embedding features with ESM C. In script, you need to modify the file paths according to your needs.Run the following commands:
<pre> python ESM_embedding.py </pre>
This will generate the corresponding embedding features files.

Next, you can proceed with model training and validation using the provided `Train.py` script. Before running it, make sure you have prepared the training features and modified the file paths and other parameters according to your needs. Run the following command to start the model training and validation:
<pre> python Train.py </pre>
After the script finishes running, it will generate the best model file for each fold in the five-fold cross-validation, as well as the final saved model file.

Finally, you can test the model on a independent dataset using the provided `Test.py` script. Make sure you have prepared the test features and modified the file path of saved model according to your needs. Run the following command to test the model:
<pre> python Test.py </pre>

##  Dataset

The benchmark dataset and the generalization dataset used in this study is included in the `Dataset` folder.

##  Concat

If you have any questions, suggestions, or collaboration interests, please feel free to reach out:

**Email:** kasval183@gmail.com
