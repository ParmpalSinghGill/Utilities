sudo docker build  -t docker-tf-keras-deploy .
sudo docker run -p 8080:8080 --rm docker-tf-keras-deploy serve

