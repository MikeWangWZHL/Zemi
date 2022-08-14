

FROM mirrors.tencent.com/ai-lab-seattle/kic-t0:latest

# ENV http_proxy="http://us-sg2-devproxy-prod.oa.tencent.com:8080" \
#     ftp_proxy="http://us-sg2-devproxy-prod.oa.tencent.com:8080" \
#     https_proxy="http://us-sg2-devproxy-prod.oa.tencent.com:8080"
# ENV HF_HOME=/cache/huggingface

RUN pip install einops
RUN pip install einops-exts
RUN pip install -U transformers==4.19.4
