FROM python:3
ADD video_screening.py /
ADD models /
RUN pip install cv2
RUN pip install smtplib
RUN pip install PIL
CMD [ "python", "./video_screening.py" ]