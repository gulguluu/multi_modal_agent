FROM intel/intel-extension-for-pytorch:2.7.10-serving-xpu

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

RUN mkdir -p /app/data

ENV IPEX_OPTIMIZE=1
ENV IPEX_MERGE_FUSION=1
ENV IPEX_AUTO_TUNE=1

CMD ["python", "main.py"]
