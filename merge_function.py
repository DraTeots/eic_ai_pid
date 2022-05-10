import numpy as np

# def merge_data(inputs, answers):
#      return merged_inputs, merged_answers 

def filter_data(inputs, answers, events_to_read):

    input_first = inputs[:events_to_read]
    input_second = inputs[events_to_read:]
    input_merged = np.add(input_first, input_second)

    answers_first = answers[:events_to_read]
    answers_second = answers[events_to_read:]
    answers_merged = np.add(answers_first, answers_second)

    inputs = input_merged
    answers = answers_merged

    # filtering events that are within 1 square of each other
    tf = []
    for i in range(len(answers_first)):
        col1 = np.argmax(np.argmax(answers_first[i], axis=1)) - 1
        row1 = np.argmax(np.argmax(answers_first[i], axis=0)) - 1

        col2 = np.argmax(np.argmax(answers_second[i], axis=1)) - 1
        row2 = np.argmax(np.argmax(answers_second[i], axis=0)) - 1

        if np.abs(col1 - col2) < 2 and np.abs(row1 - row2) < 2:
            tf.append(False)
        else:
            tf.append(True)
    tf = np.array(tf)

    inputs = inputs[tf]
    answers = answers[tf]
    return inputs, answers
    