CMCL 2022 Shared Task

Evaluation Phase: Test data - Subtask 2

The file test_data.csv contains 402 sentences in a new language that will be used to evaluate your models on.
The new language is Danish (da).
The contents are in the following format: language, sentence_id,word_id,word

Your model should predict the features FFDAvg, FFDStd, TRTAvg, TRTStd.

To make a submission, you need to upload a ZIP file containing a text file named "answer.txt", which contains your predictions in the same format as the training data. This means a header line containing "language,sentence_id,word_id,word,FFDAvg,FFDStd,TRTAvg,TRTStd", followed by one token per line as in the test data file including the predicted feature values.

Here are the first few lines of an answer.txt file as an example:

language,sentence_id,word_id,word,FFDAvg,FFDStd,TRTAvg,TRTStd
da,CopCo1-25,15,hvid,6.15,2.42,16.07,8.61
da,CopCo1-25,16,jul.,7.54,1.23,7.59,2.11

During the evaluation phase you can test your submissions against the real eye-tracking features. You are allowed to make a total of 3 submissions.

If you have any questions, contact cmclsharedtask@gmail.com.
