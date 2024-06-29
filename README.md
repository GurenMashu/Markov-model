# Markov Model

## Markov Model Classifier

The poem files used were the ones provided by Lazy Programmer Inc in his Udemy course on NLP.

A simple project of making a Markov model classifier during my studies in NLP. This was done as a project that was a part of a Udemy course by Lazy Programmer Inc. This is the first of 2 projects.

### Usage

Download the .txt files and then copy their path to paste in the program.

#### For both projects, it's important to change the directory of the data on which the model is trained. This is important as the code won't work otherwise. Make sure that the file path is set correctly in the appropriate list.

### Model Performance

*  The average recall is 100.00%.
*  The average F1 score is 79.80%.
*  The average precision is 66.42%.

### Conclusion of Project 1

The recall value of 100% indicates the model is very capable of capturing true positives; however, the lower value of precision indicates 33.58% of all positives are falsely matched.

The model clearly needs improvement, specifically in terms of reducing false positives, which can potentially be rectified by improving the training data as the false positives might be arising due to ambiguous text where both authors are equally likely.

## Markov Model Text Generator
The poem file used was provided by Lazy Programmer Inc in his Udemy course on NLP. This is the second part of two projects where I create a text generator.

### Usage
Download one of two poem files provided in the repo and paste the path into the Markov text generator program in the appropriate place.

Simply run the program, and a random sequence of words (with 'some' congruence to comprehensible text) is generated.

### Conclusion of Project 2
The generation is not the best, but it sometimes makes 'poetic phrases', so all in all, a good model for having a laugh.

## Conclusion
This was just a learning experience, and the classifier model, as is evident, didn't fare too well in terms of precision compared to most modern algorithms for classifying text. The generation model fared not too badly in making sentences. Though mostly meaningless, it does provide on occasion a snippet of meaningful lines.
