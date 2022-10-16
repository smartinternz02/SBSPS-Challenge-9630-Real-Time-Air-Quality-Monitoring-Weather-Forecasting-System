from django.shortcuts import render
import urllib.request
import json
import joblib
import numpy as np
def home(request):
    return render(request,'index.html')

def weather(request):

    if request.method == 'POST':
        city = request.POST['city']

        source = urllib.request.urlopen('http://api.openweathermap.org/data/2.5/weather?q=' +
                                        city + '&units=metric&appid=a0a23c127804518f75e75713d0ddb52c').read()
        list_of_data = json.loads(source)

        data = {
            "country_code": str(list_of_data['sys']['country']),
            "coordinate": str(list_of_data['coord']['lon']) + ', '
            + str(list_of_data['coord']['lat']),
            
            "temp": str(list_of_data['main']['temp']) + ' °C',
            "pressure": str(list_of_data['main']['pressure']),
            "humidity": str(list_of_data['main']['humidity']),
            'main': str(list_of_data['weather'][0]['main']),
            'description': str(list_of_data['weather'][0]['description']),
            'icon': list_of_data['weather'][0]['icon'],
        }
        print(data)
    else:
        data = {}

    return render(request, "weatherforecasting.html", data)

def result(request):

    return render(request,"result.html")#,{'AQI:':answer,'AQI Comment:':ans})


def aqi_result(request):
    rf_random = joblib.load('final_model.sav')
    lis = [] #14.6,0,2.1,23.4,67.8,9.6,9,0,8.32

    lis.append(request.POST.get('a'))
    lis.append(request.POST.get('b'))
    lis.append(request.POST.get('c'))
    lis.append(request.POST.get('d'))
    lis.append(request.POST.get('e'))
    lis.append(request.POST.get('f'))
    lis.append(request.POST.get('g'))
    lis.append(request.POST.get('h'))
    lis.append(request.POST.get('i'))

    print(lis)

    #v = np.array(lis, dtype=int)
    #val = [int(i) for i in v]
    answer = rf_random.predict([lis])[0]

    if answer <=50:
        ans = "Good"
    elif answer>50 and answer <=100:
        ans = "Moderate"
    elif answer>100 and answer <=150:
        ans = "Satisfactory"
    elif answer>150 and answer <=200:
        ans = "Poor"
    elif answer>200 and answer <=300:
        ans = "Very Poor"
    elif answer>300 and answer <=400:
        ans = "Severe"
    else:
        ans = "Hazardous"

    
    #return render(request,"result.html")#,{'AQI:':answer,'AQI Comment:':ans})

    return render(request,'aqi_result.html',{'answer':answer,'ans':ans})

    