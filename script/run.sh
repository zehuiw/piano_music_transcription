python ../Preprocessing/WAV2mat_batch.py ../script/input_wave_file.txt
# Generate 0input_wave_file_X.npy, 0input_wave_file_y.npy
python ../Preprocessing/mat2norm_single.py ./
# Normalize X.npy and replace it
python predict.py ./ ../weights/0307_06model_07data/ cnn

python npy2midi.py 0input_wave_file_y.npy ground_truth
python npy2midi.py predictions.npy output