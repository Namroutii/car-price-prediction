from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('PolynomialRegressionModel.pkl','rb'))
cars = pd.read_csv('./data/CSV/cleaned_data.csv')

@app.route('/',methods=['GET','POST'])
def index():
    Car = sorted(cars['Car'].unique())
    Year = sorted(cars['Year'].unique(),reverse=True)
    Color = cars['Color'].unique()
    Fuel = cars['Fuel'].unique()
    Origin = cars['Origin'].unique()
    License = cars['License'].unique()
    Gear = cars['Gear'].unique()
    Glass = cars['Glass'].unique()
    Power = cars['Power'].unique()
    Meter = cars['Meter'].unique()
    Passengers = cars['Passengers'].unique()
    Payment = cars['Payment'].unique()
    Previous_owners = cars['Previous_owners'].unique()
    Condition = cars['Condition'].unique()
    Central_locking = cars['Central_locking'].unique()
    Alarm = cars['Alarm'].unique()
    Radio = cars['Radio'].unique()
    Sunroof = cars['Sunroof'].unique()
    Magnesium_wheels = cars['Magnesium_wheels'].unique()
    Leather_seats = cars['Leather_seats'].unique()
    Airbag = cars['Airbag'].unique()
    For_sale = cars['For_sale'].unique()
    For_exchange = cars['For_exchange'].unique()

    return render_template('index.html',Car = Car,Year=Year,Color=Color,Fuel=Fuel,Origin=Origin,License=License,Gear=Gear,Glass=Glass,Power=Power,Meter=Meter,Passengers=Passengers,Payment=Payment,Previous_owners=Previous_owners,
    Condition=Condition,Central_locking=Central_locking,Alarm=Alarm,Radio=Radio,Sunroof=Sunroof,Magnesium_wheels=Magnesium_wheels,Leather_seats=Leather_seats,Airbag=Airbag,For_sale=For_sale,For_exchange= For_exchange)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    Car = request.form.get('Car')
    Year = request.form.get('Year')
    Color = request.form.get('Color')
    Fuel = request.form.get('Fuel')
    Origin = request.form.get('Origin')
    License = request.form.get('License')
    Gear = request.form.get('Gear')
    Glass = request.form.get('Glass')
    Power = request.form.get('Power')
    Meter = request.form.get('Meter')
    Passengers = request.form.get('Passengers')
    Payment = request.form.get('Payment')
    Previous_owners = request.form.get('Previous_owners')
    Condition = request.form.get('Condition')
    Central_locking = request.form.get('Central_locking')
    Alarm = request.form.get('Alarm')
    Radio = request.form.get('Radio')
    Sunroof = request.form.get('Sunroof')
    Magnesium_wheels = request.form.get('Magnesium_wheels')
    Leather_seats = request.form.get('Leather_seats')
    Airbag = request.form.get('Airbag')
    For_sale = request.form.get('For_sale')
    For_exchange = request.form.get('For_exchange')


    prediction= model.predict(pd.DataFrame(columns=['Car','Year','Color','Fuel','Origin','License','Gear','Glass','Power','Meter','Passengers','Payment','Previous_owners','Condition','Central_locking','Alarm','Radio','Sunroof','Magnesium_wheels','Leather_seats','Airbag','For_sale','For_exchange'],
                        data=np.array([Car,Year,Color,Fuel,Origin,License,Gear,Glass,Power,Meter,Passengers,Payment,Previous_owners,Condition,Central_locking,Alarm,Radio,Sunroof,Magnesium_wheels,Leather_seats,Airbag,For_sale,For_exchange])
                        .reshape(1, -1)))

    print(prediction)

    return str(int(np.round(prediction[0])))





if __name__ == '__main__':
    app.run(debug=True)
