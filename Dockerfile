FROM python:3.6.8
ADD video_screening.py /
ADD models /
RUN pip install image
RUN pip install tensorflow
RUN pip install opencv-python
RUN pip install keras
RUN pip install matplotlib
RUN pip install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl
CMD [ "python", "./video_screening.py" ]