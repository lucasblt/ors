FROM python:3.8

RUN apt-get update -y && apt-get install -y

COPY webapp/app/requirements.txt /app/requirements.txt

WORKDIR /app

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install -r requirements.txt

COPY . /app

EXPOSE 1234

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]