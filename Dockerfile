FROM python:3
ADD video_screening.py /
CMD [ "python", "./video_screening.py" ]