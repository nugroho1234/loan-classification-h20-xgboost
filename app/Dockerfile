FROM ubuntu

RUN apt update -y
RUN apt install python3 -y --fix-missing
RUN apt install python3-pip -y
RUN apt install openjdk-8-jre-headless -y

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY model /app/model
COPY main.py preprocess.py loan_classification_variables.py ./

EXPOSE 5000

CMD ["uvicorn", "main:LoanPredApp", "--reload", "--host", "0.0.0.0", "--port", "5000"]
