FROM python:3.10-slim
WORKDIR /app
RUN pip install paho-mqtt==1.6.1
COPY iot_device.py .
CMD ["python", "-u", "iot_device.py"]