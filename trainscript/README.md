In order to train models using scripts contained here, you should do the following things.

1. Compile a customized matconvnet contained in '[../libs/matconvnet-1.0-beta23_modifiedDagnn](https://github.com/aimerykong/Dimensional-Emotion-Analysis-of-Facial-Expression/tree/master/libs/matconvnet-1.0-beta23_modifiedDagnn)'.

2. Download all files from [google drive](https://drive.google.com/open?id=1W26vVDWWMgKQGxYx7y0ljng8E-Vb2LPY), and put them under directory '[./transcript/exp/](https://github.com/aimerykong/Dimensional-Emotion-Analysis-of-Facial-Expression/tree/master/trainScript/exp)'.

3. Download the dataset from [google drive](https://drive.google.com/open?id=1s79cTqa9ftVfynUk0uZdQZUElozsaQ6l), and put it here under current directory.



Once the above are done, you are good to go -- 

1. Run scripts with prefix '_main00X_' will train models accordingly as the file name suggests.

2. Run scripts with suffix '_eval.m_' will evaluate the model specified in the script.

3. Run scripts with name prefix '_main005_' will fine-tune a three-dimenional model for the corresponding two-dimensional model.




Shu Kong @ UCI
5/23/2018
