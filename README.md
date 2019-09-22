# Disease-Predictor-with-ChatBot-feature
Built a Machine Learning Model to detect the disease given the symptoms.
The Model was trained using Random Forest Classifier and got an accuracy of 1.0

The ChatBot feature is present in the model which takes the symptoms as input provide the Disease as output
Also Google Search was embedded to get the treatment of the disease.

There is 2 levels of symptoms checking:

(1) Less questions were asked and only relatable questions were asked to the user. In case he/she is not satisfied with the result then it goes to next level symptom checking (Step 2).
Else it directly goes to Step 3.

(2) Here every possible symptom was asked and then the disease was predicted.

(3) Here it open your google chrome browser and searches for the treatment/remedy for the disease predicted.

Currently taking inputs randomly, to give user input just comment out the random input lines and uncomment the user input lines.
