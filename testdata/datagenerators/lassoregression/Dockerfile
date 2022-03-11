FROM python:3

#install sklearn, pandas and numpy
RUN pip install pandas && pip install -U scikit-learn && pip install numpy
#copy the script code to the image
COPY ./testdata/datagenerators/lassoregression/script /generator
# enter the generator directory in the image
WORKDIR /generator
# run the python script to generate the test data
ENTRYPOINT ["python", "script.py"]