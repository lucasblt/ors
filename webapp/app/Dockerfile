FROM python:3.8

RUN apt-get update -y && apt-get install -y

COPY ./webapp/app /app

WORKDIR /app

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install -r requirements.txt

EXPOSE 1234

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]