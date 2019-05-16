Read Me File :

1. Datasets used : "Facades" and "cityscapes"

2. Running the code
   Assuming the dataset is stored in correct format(./$dataset_name/train/ and ./$dataset_name/test/).
   Running the first 2 cells for downloading and extracting data will take care of the format
	a. specify the dataset($dataset_name) to be used when creating an instance of the CGAN model?
	b. call the train function on the instance
	c.the $start parameter is used for resuming running from saved weights
   the train function will save weights for every epoch and generate validation images using test set every $sample_interval
   the validation images will be saved in ./images/$dataset_name/*

*This code was run and tested on google colab environment.