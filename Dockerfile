FROM python:3
ADD video_screening.py /
ADD models /
RUN pip install opencv-python
CMD [ "python", "./video_screening.py" ]