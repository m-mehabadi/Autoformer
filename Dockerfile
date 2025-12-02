FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

COPY Autoformer/ /workdir/Autoformer/

WORKDIR /workdir/Autoformer

RUN pip3 install --no-cache-dir -r ./requirements.txt

ENTRYPOINT ["python3", "quick_test.py"]


