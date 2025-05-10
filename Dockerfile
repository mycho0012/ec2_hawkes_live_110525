FROM python:3.9-slim

WORKDIR /app

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 필요한 파일 복사
COPY requirements.txt .
COPY hawkes.py .
COPY ec2_hawkes_live.py .
COPY fix_pandas_ta.py .

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# pandas_ta 패치 적용
RUN python fix_pandas_ta.py

# 디렉토리 생성
RUN mkdir -p /app/logs

# 타임존 설정 (한국 시간)
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 컨테이너 시작 명령
CMD ["python", "ec2_hawkes_live.py"]