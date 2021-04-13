from context import scripts
import scripts
from scripts.load_GENDER import get_data
from keras.models import load_model


model1 = load_model('baselneLSTM.h5')
loss, accuracy = model1.evaluate(matrix_test_x, y_test, verbose=0)
print(f'Baseline NN - Accuracy: {accuracy*100}%')


model2 = load_model('classicFeedFWD.h5')
loss, accuracy = model2.evaluate(matrix_test_x, y_test, verbose=0)
print(f'Classic NN - Accuracy: {accuracy*100}%')


model2 = load_model('customLSTM.h5')
loss, accuracy = model3.evaluate(matrix_test_x, y_test, verbose=0)
print(f'Best Custom NN - Accuracy: {accuracy*100}%')



