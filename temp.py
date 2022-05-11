from flask import Flask,render_template,request
import joblib

temp = Flask(__name__)
model_Min=joblib.load('ai/modelmin.h5')
scaler_min=joblib.load('ai/scalermin.h5')
model_Max=joblib.load('ai/modelmax.h5')
scaler_max=joblib.load('ai/scalermax.h5')

@temp.route('/',methods=['GET'])
def index():
    return render_template('temp.html')

@temp.route('/predict',methods=['GET'])
def predict_data():
    min_Hum=float(request.args['Minimum_hum'])
    max_Hum=float(request.args['Maximum_hum'])
    max_Temp=float(request.args['Maximum_temp'])
    min_Temp=float(request.args['Minimum_temp'])
    WS=float(request.args['wind speed'])
    LHF=float(request.args['latent Heat'])
    CC1=float(request.args['cloud cover (0-5 h)'])
    CC2=float(request.args['cloud cover (6-11 h)'])
    CC3=float(request.args['cloud cover (12-17 h)'])
    CC4=float(request.args['cloud cover (18-23 h)'])
    PPT1=float(request.args['precipitation (0-5 h)'])
    PPT2=float(request.args['precipitation(6-11 h)'])
    PPT3=float(request.args['precipitation (12-17 h)'])
    PPT4=float(request.args['precipitation (18-23 h)'])
    lat=float(request.args['lat'])
    lon=float(request.args['lon'])
    solar_radiation=float(request.args['Solar radiation'])

    data=[min_Hum,max_Hum,max_Temp,min_Temp,WS,LHF,CC1,CC2,CC3,CC4,PPT1,PPT2,PPT3,PPT3,lat,lon,solar_radiation]
    data=scaler_min.transform([data])
    minimum_Prediction=round(model_Min.predict(data)[0])
    maximum_Prediction=round(model_Max.predict(data)[0])
    return render_template('temp.html',minimum_Prediction=minimum_Prediction,maximum_Prediction=maximum_Prediction)
 
if __name__ == '__main__':
    temp.run()