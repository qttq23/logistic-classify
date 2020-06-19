
# requirements:
1. anaconda3 installed
2. libraries: numpy, pandas, matplotlib, PIL, sklearn, pickle, seaborn

# run:
1. open anaconda3 and navigate to this folder
2. type:
python train.py <path_to_train_data> <path_to_save_model>
python test.py <path_to_model> <path_to_test_data> <path_to_save_prediction>

eg:
python train.py res\train.csv res\output.model
python test.py res\output.model res\train.csv output\ypred.csv
