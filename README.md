# Aspect-Weighted Sentiment Analysis with Ontology and Deep Learning

## Description:
On a restaurant dataset of 3044 restaurant reviews, we perform sentiment analysis.
1. The task is to determine sentiment polarity of each review.
2. We need to consider the aspects mentioned in each review along with their sentiments.
3. We calculate the **_weight(importance)_** of each aspect.
4. In sentences where multiple aspects are mentioned, these weights are used to determine the importance of each aspect's sentiment.
5. Finally, overall sentiment is calculated from these individual sentiment polarities.

We use a deep neural network (LSTM) where reviews are fed into the model as word embeddings. Information about aspect weights(importance) (for the aspects present in the review) are added to these word embeddings. Output of the model is positive or negative (1/0) denoting the overall sentiment of a review.

## Dataset:
> Original dataset: *./dataset/Restaurants_Train_v2.xml*.

We use an open-source dataset of SemEval-2014 competition (Task 4) (http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)

* There were 3044 restaurant reviews on which we perform sentiment analysis.
* The dataset contains reviews for restaurants along with aspect level sentiment polarities.
* Each review may or may not consist of one or more of the following aspect categories:
    a) food, b) service, c) price, d) ambience, e) anecdotes/miscellaneous.
* Each review in the dataset contains the aspects that have been mentioned in the review text. Sentiment expressed for each aspect category is also mentioned along with each of them.
* In each review, there are certain words which map to one of the above mentioned five aspect categories. These words are considered as aspect terms and are also separately mentioned along with each review. Along with each of these aspect terms, sentiment expressed for it is also mentioned. 

> Annotated dataset: *./test_data.xml*

We manually annotate the dataset, by augmenting each review with overall sentiment (positive or negative).

**Annotation guidelines**
* There are 5 aspectCategories: food, service, price, ambience and anecdotes/miscellaneous (in order of importance of the aspects, highest to lowest). This priority is decided on common world knowledge.
* For every sentence, annotation needs to be done for overall sentiment polarity.
* We manually annotate the dataset, by augmenting each review with overall sentiment (positive or negative).

## Workflow
1. Run the _aspect_weights_cpt.py_ file first. This code converts xml data to csv data and also calculates the weights of the aspects.
2. Run the model.py to get the final predictions and the accuracy, precision, etc measures.