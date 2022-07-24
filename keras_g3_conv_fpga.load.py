import os
import numpy as np
from geant3_parser import build_true_answers_train_set
from geant3_parser import Geant3DataFile
from keras.models import load_model


def norm_func(e):
    return np.float64(np.log(e) / 11)


file_name = os.path.join('data', 'shower_geant3_new.dat')
data_file = Geant3DataFile(file_name, skip_lines=3)
input_data, answers, values = build_true_answers_train_set(data_file, 50000, norm_func=norm_func, rnd_shift=((-2,2), (-2,2)))

# Reshape to events x 11 x 11 x 1 form
input_data = np.reshape(input_data, (len(input_data), 11, 11, 1))  # -1 => autodetermine
answers = np.reshape(answers, (len(answers), 11, 11, 1))  # -1 => autodetermine
# printout shape
print(f"Inputs shape new = {np.shape(input_data)}")
print(np.shape(answers))
print(np.shape(input_data))


# Load NN model
model = load_model(os.path.join("trained_models", f"g3_conv_bce-sgm_fpga_2dense_700000.hd5"))

# print summary
model.summary()

# Run ML on data
ml_events = model.predict(input_data)

# reshape events
ml_events = np.reshape(ml_events, (len(ml_events), 11, 11, 1))