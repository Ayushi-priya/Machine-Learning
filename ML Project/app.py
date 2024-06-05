from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from flask import Flask

app = Flask(__name__, template_folder='template')

#load the pickel model
with open('./RandomForestmodel.pkl', 'rb') as p:
    model = pickle.load(p)

with open('./scalar.pkl', 'rb') as p:
    scalar = pickle.load(p)

encoding = {
    'CHK_ACCT': {'0DM': 0, 'less-200DM': 1, 'no-account': 2, 'over-200DM': 3},
    'History': {'all-paid-duly': 0, 'bank-paid-duly': 1, 'critical': 2, 'delay': 3, 'duly-till-now': 4},
    'Purpose of credit': {'business': 0, 'domestic-app': 1, 'education': 2, 'furniture': 3, 'new-car': 4,
                          'others': 5, 'radio-tv': 6, 'repairs': 7, 'retraining': 8, 'used-car': 9},
    'Balance in Savings A/C': {'less1000DM': 0, 'less100DM': 1, 'less500DM': 3, 'over1000DM': 4, 'unknown': 5},
    'Employment': {'four-years': 0, 'one-year': 1, 'over-seven': 2, 'seven-years': 3, 'unemployed': 4},
    'Marital status': {'female-divorced': 0, 'male-divorced': 1, 'married-male': 2, 'single-male': 3},
    'Co-applicant': {'co-applicant': 0, 'guarantor': 1, 'none': 2},
    'Real Estate': {'building-society': 0, 'car': 1, 'none': 2, 'real-estate': 3},
    'Other installment': {'bank': 0, 'none': 1, 'stores': 2},
    'Residence': {'free': 0, 'own': 1, 'rent': 2},
    'Job': {'management': 0, 'skilled': 1, 'unemployed-non-resident': 2, 'unskilled-resident': 3},
    'Phone': {'no': 0, 'yes': 1},
    'Foreign': {'no': 0, 'yes': 1}
}

@app.route("/", methods=['GET', 'POST'])
def predict():
  prediction="The Prediction is..."
  custom_data_input_dict={}
  
  if request.method=='POST':
    #Collect data from form and store in dictionary format
    for key, x in request.form.items():
      try:
         x=int(x)
      except:
         pass
      
      custom_data_input_dict[key] = x

    #Mapping the collected data value into Label encoded values of the trained model
    for key, value in encoding.items():
      custom_data_input_dict[key] = value[custom_data_input_dict[key]]

    #Creating a DataFrame of the collected data
    features = pd.DataFrame([custom_data_input_dict])

    #Standardize the features
    data_scaled = scalar.transform(features)

    #Predict result 
    prediction = model.predict(data_scaled)

    if prediction == [0]:
       prediction="The Prediction is Bad"
    elif prediction == [1]:
       prediction="The Prediction is Good"

  return render_template("index.html", results=prediction)

if __name__ == '__main__':
    app.debug = True
    app.run()





