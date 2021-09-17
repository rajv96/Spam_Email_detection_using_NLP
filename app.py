from flask import Flask, render_template ,url_for, request
import pandas as pd 
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle
import string
from nltk.corpus import stopwords

def text_process(mess):
    
    """Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    
    #check characters to see if there are any punctuations
    nopunc = [char for char in mess if char not in string.punctuation]
    
    #join characters to form a string
    nopunc = ''.join(nopunc)
    
    #remove the stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# load the model from disk
filename = 'model.pkl'
pipeline_RF = pickle.load(open(filename, 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		my_prediction = pipeline_RF.predict(data)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':  
	app.run(debug=True)