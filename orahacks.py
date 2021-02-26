from flask import Flask, render_template, request

from flask_restful import Api, Resource

from cdqa.pipeline import QAPipeline
import pandas as pd
import pickle
import jsonify
import re

app = Flask('Customer Warriors') 



dataframe_from_pkl = pd.read_pickle('./csv_of_df_scm.pkl')

with open('urldict.pickle', 'rb') as handle:
    url_dict = pickle.load(handle) 

model=QAPipeline(reader='./distilbert_qa_finetuned.joblib', max_df=1.0)
model.fit_retriever(df=dataframe_from_pkl)

def show_predictions(pred, url_dict):
    return  (pred[0]) , (url_dict.get(pred[1])), (pred[2])

@app.route('/')
@app.route( '/index.html')
def home():
    return render_template('index.html')


@app.route('/SomeSampleQnAs.html')
def show_sample_qnas():
    return render_template('SomeSampleQnAs.html')
@app.route('/LoveQnA.html')
def show_love_qnas():
    return render_template('LoveQnA.html')

@app.route('/KPIsQnA.html')
def show_kpis_qnas():
    return render_template('KPIsQnA.html')


@app.route('/YourQuestion.html')
def show_predict_paragraph():
    return render_template('YourQuestion.html')
@app.route('/Team.html')
def show_team():
    return render_template('Team.html')
@app.route('/SourceDocs.html')
def show_list_of_docs():
    return render_template('SourceDocs.html')
@app.route('/Results.html', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':

        inputsentence = request.form['inputword']
 
        prediction = model.predict(inputsentence, n_predictions=3)

        short_answer1 , doc_link1, paragraph1 = show_predictions(prediction[0],url_dict)
        short_answer2 , doc_link2, paragraph2 = show_predictions(prediction[1],url_dict)
        short_answer3 , doc_link3, paragraph3 = show_predictions(prediction[2],url_dict)
        
        

        def resolve_link(doc_link, paragraph):
            regex_string=re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)|$', re.DOTALL)  
            if doc_link is None:
                link_from_paragraph=re.search(regex_string, paragraph).group()
                if link_from_paragraph:
                    doc_link=link_from_paragraph
                else:
                    doc_link='https://mrg-ai.github.io/blog/2021/02/26/Customer-Warriors-Orahacks.html'
            return doc_link

        doc_link1 = resolve_link(doc_link1,paragraph1)
        doc_link2 = resolve_link(doc_link2,paragraph2)
        doc_link3 = resolve_link(doc_link3,paragraph3) 


        return render_template('Results.html', inputword=inputsentence
		,   
        short_answer1=short_answer1, doc_link1= doc_link1, paragraph1=paragraph1,
        short_answer2=short_answer2, doc_link2= doc_link2, paragraph2=paragraph2,
        short_answer3=short_answer3, doc_link3= doc_link3, paragraph3=paragraph3
		) 
    
api = Api(app)

class BotX(Resource):

    def get(self,search_txt):

        #inputsentence = request.form['inputword']

        prediction = model.predict(search_txt, n_predictions=3)

        short_answer1 , doc_link1, paragraph1 = show_predictions(prediction[0],url_dict)
        short_answer2 , doc_link2, paragraph2 = show_predictions(prediction[1],url_dict)
        short_answer3 , doc_link3, paragraph3 = show_predictions(prediction[2],url_dict)

        results_dict ={
                          "short_answer1": [short_answer1],
                          "doc_link1": [doc_link1],
                          "paragraph1": [paragraph1],    
                          "short_answer2": [short_answer2],
                          "doc_link2": [doc_link2],
                          "paragraph2": [paragraph2],
                          "short_answer3": [short_answer3],
                          "doc_link3": [doc_link3],
                          "paragraph3": [paragraph3]
                      }

        
        return(results_dict)
    

api.add_resource(BotX, "/api/<string:search_txt>")


app.run("den00pkj.us.oracle.com", "9999", debug=True)
