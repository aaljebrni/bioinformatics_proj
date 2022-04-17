This is the code I wrote and used for the project of the Bioinformatics class

how to get the final results?
you can directly run either bioinformatics_project_model_1DCNN_classifier.py or bioinformatics_project_model_DNN_classifier.py and they will consume the attached csv files!

the reduced datasets are in the following csv files:
reduced_ds_256_sm.csv  //dimensionality reduced using the DAE
pca_reduced_ds_256.csv //dimensionality reduced using PCA
kpca_reduced_ds_256.csv //dimensionality reduced using KPCA


the E-TABM-185-sdrf1.csv file is only used when u wanna create a new dataset using the code in bioinformatics_project_dataset_creation.py, of course u will have to use the microarray file as well, I didn't attach it here because of its massive size, and then use the DAE to reduce the dimensionality of the generated dataset. In case you did not find any attached csv files, except the 'E-TABM-185-sdrf1.csv', you can run the files in the following order (assuming you have the microarray file):
1-bioinformatics_project_dataset_creation.py
2-bioinformatics_project_autoencoder.py
 then one of the classifiers:
bioinformatics_project_model_1DCNN_classifier.py, or bioinformatics_project_model_DNN_classifier, these classifiers will consume the generated csv files from 1 & 2


Thanks
