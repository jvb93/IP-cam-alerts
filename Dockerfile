FROM python:3
ADD video_screening.py /
ADD models /
RUN pip install opencv-python
RUN pip install imageai
CMD [ "python", "./video_screening.py" ]