FROM python:3
ADD video_screening.py /
ADD models /
pip install cv2
pip install smtplib
pip install PIL
CMD [ "python", "./video_screening.py" ]