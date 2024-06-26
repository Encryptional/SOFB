# SOFB is a Comprehensive Ensemble Deep Learning approach for Elucidating and Characterizing Protein-Nucleic-Acid-Binding Residues  
This repository is developed for nucleic acid binding residues identification using SOFB, which implemented an ensemble deep learning model-based sequence network.  

# Requirement  
SOFB is developed under Linux environment with:  
python  3.8.0
transformers  4.8.2  
torch  1.8.1  
tensorflow  2.7.0  
sentencepiece 0.1.96  
protobuf  3.19.1

# Bioinformatics tools and database   
To run the SOFB, you need to install the bioinformatics tools and download the corresponding databases:  
(1) Install blast+ for extracting PSSM(position-specific scoring matrix) profiles  
To install ncbi-blast-2.8.1+ and download NR database (ftp://ftp.ncbi.nlm.nih.gov/blast/db/) for psiblast, please refer to BLAST(https://www.ncbi.nlm.nih.gov/books/NBK52640/).  
(2) Install HHblits for extracting HMM profiles
To install HHblits and download uniclust30_2018_08 (http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz) for HHblits, please refer to https://github.com/soedinglab/hh-suite.


# Extract multi-features using various method  
For the PKs, Physicochemical characteristics and RAA in DNA and RNA, you can generate different results by changing the datasets path and the save file name in the file:  
```
python Pks.py 
python RAA.py 
python pychar.py 
```
In this way, you can get (raa_train; pychar_train1; pychar_train2; pychar_train3; pkx_train) and (raa_test; pychar_test1; pychar_test2; pychar_test3; pkx_test).  

After that, you can get other features by run the command, particularly, the model used for generating dynamic embedding is provided in fighsare(https://figshare.com/articles/online_resource/SOFB_figshare_rar/25499452), you need download it and set the path in the program :  
```
python generate_multi_feature.py --nucleic_acid RNA(or DNA)
```
Then, you can get other bio-features (train_one_hot; train_bio_PSSM; train_bio__HMM; test_one_hot; test_bio_PSSM; test_bio_HMM) and dynamic language embeddings(NABert_train; NABert_test; ProtT5_train; ProtT5_test).  

Particularly, all the bio-features will concatenate as(train_bio_vec; test_bio_vec).
# Prediction and test 
After getting the all data_vec(train_T5, train_bio_vec, train_NABert; test_T5, test_bio_vec, test_NABert), you can train a new model (or make the test) by the train (test) function in  'predict.py' file by the command:  
```
python predict.py --nucleic_acid RNA --epochs 30 --batchsize 1024 --ensemble 4
```

# Interpretability  and Visualization
The interpretability analysis of our model refers to the SHAP(https://github.com/slundberg/shap) and the Visualization refers t othe ProtTrans(https://github.com/agemagician/ProtTrans).

<p align="center">
  <img width="600" height=800 src="intepretability.png">
  <p align="center">intepretability of the SOFB</p><br><br>
</p>
  
<p align="center">
  <img width="700" height=800 src="visualization.png">
  <p align="center">visualization of the bio-language learning model</p><br><br>
</p>
