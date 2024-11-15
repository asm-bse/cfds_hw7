import requests

url = "http://127.0.0.1:8000/predict"
data = [
    {
        "age": 45,
        "height": 170,
        "weight": 70,
        "aids": 1,
        "cirrhosis": 0,
        "hepatic_failure": 0,
        "immunosuppression": 1,
        "leukemia": 0,
        "lymphoma": 0,
        "solid_tumor_with_metastasis": 1
    }
]
response = requests.post(url, json=data)
print(response.json())
