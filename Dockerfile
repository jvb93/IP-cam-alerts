FROM python:3
ADD video_screening.py /
ADD models /
CMD [ "python", "./video_screening.py" ]