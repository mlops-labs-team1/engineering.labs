FROM python:3.8.1-slim

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt

# add app
COPY . /usr/src/app

ENTRYPOINT ["flask", "run"]
CMD ["--host=0.0.0.0", "--port=5000"]