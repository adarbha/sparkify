# Sparkify


## Table of Contents
1.  Installation <br/>
    The code in notebook Sparkify.ipynb is run on Python 3.6.x. Please install all the python dependencies using **requirements.txt**. Notebook is run on Spark in local mode. Hence all pyspark dependencies need to be installed for the code to execute without hiccups

2. Project Motivation <br/>
	Sparkify is an online music streaming service, like Spotify, where users can play music on demand or in a random mode, based on their subscription parameters. Data logs are collected on every action performed by every user and is compiled into structured json file. The goal of this project is to predict churn of a customer by engineering relevant features using the data provided. For this analysis, churn is defined as a user who is canceling his/her subscription. User downgrade is one of the potential ways to define churn, but this aspect is not explored in this analysis.
	 
    
3. Files and their description
	- README.md - This file
	- Sparkify.ipynb - Jupyter notebook where a classification model is built on a smaller dataset mini_sparkify_event_data.json. This code acts as a basis for building the script file uploaded to Amazon EMR cluster for building a data pipeline and runnning the model on full dataset
	- Sparkify.py - This is a script file that is transfered to AWS EMR cluster to train the model on 12GB sparkify dataset. Script file is develped using the notebook as a prototype
	- mimi_sparkify_event_data.json.zip - Compressed file of the smaller json mentioned above
	- requirements.txt - Python dependencies for running the notebook
	- license.txt - MIT license file

   
    
4. Results <br/>
	From sparkify notebook run on mini_sparkify dataset
	- From printSchema, it has been identified that page column records the events in the dataset. There are 52 Cancellations in the dataset and these users have been assigned to churned users group
	- Following features have been engineered as they showed correlations with the churn patterns
		- Paid/free -  paid users have a higher affinity to leave
		- Ads served - on an avg a churned user is shown lesser number of ads on an average
		- Number of songs played - churned user on average listens to much less number of songs
		- Songs added to playlist - churned users add lesser number of songs to the playlist
		- Adding friends - churned users are less socially active on the app, adding fewer friends
		- Accessing help and settings - churned users seek less help 
		- Gender - More males churned than females
		- Thumbs ratio - defined as the ratio of thumbsup to thumbsdown, unchurned users tend to like the quality of songs compared to unchurned users, based on this ratio
	- Modeling performed based on the features mentioned above. Three classification models(logistic, gradient boosting, random forest) were tested evaluating f1-score as the dataset had a class imbalance
	- Gradient Boosting resulted in the best f1-score of 0.72
	
	From script file run full dataset
	- Crossvalidation and hyperparameter tuning using grid-based search have been performed on AWS EMR cluster
	- Search for optimal parameters by cross-validation is performed only on Gradient Boosted model as this model generated highest f1-score while prototyping in the notebook. f1-score of 0.71 is achieved on the validation dataset using this approach



5. Next steps
- Tune hyperparameters to check if we can improve on this score
- Downgrade events can be factored in to churn and a new model generated to predict such events
- Use updated version of gradient boosting, like LightGBM, to find out if we can enhance computational performance and model performance


Medium post - https://medium.com/@arjunsdarbha/customer-churn-prediction-ffc9a29852ce


### License

MIT license 
