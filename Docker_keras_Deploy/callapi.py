import requests,numpy as np

# url="http://127.0.0.1:5000/invocations"
url="http://localhost:8080/invocations"
arr = np.random.random((2, 1, 100, 121))
data = { 'arr': arr.tolist()}

response = requests.post(url, json=data)
print(response.json())