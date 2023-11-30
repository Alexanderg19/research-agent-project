FROM python:3.11

# WHere to work, and what to install to run it
WORKDIR /code

#
COPY ./requirements.txt /code/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./app /code/app

EXPOSE 8080

# Change the port number to suit your needs, we use 8081 by defaul, but choose the best port for you
CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]