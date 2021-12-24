CV 14조 final project
# AI 사진 복원 도구 : Good Bye 옥에티

## 프로젝트 소개

### 시연 영상

[![Video Label](http://img.youtube.com/vi/Mnqi91GWhiY/0.jpg)](https://www.youtube.com/watch?v=Mnqi91GWhiY)

### 서비스 사용해보기
App : [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/intelli8786/ai_blemishesremover/main/WebServer/Service.py)


## 팀원 소개


## 서비스 파이프라인

## 실행방법

### Super Resolution and Deblur module

#### Inference code & Pretrained weight Repo
* [SwinIR](https://github.com/JingyunLiang/SwinIR)
* [Deblur](https://github.com/jiangsutx/SRN-Deblur)

#### 실행 방법
1. ```streamlit run WebServer/Server_SRD.py --server.port=6006```
2. port만 변경해서 접속 (ex:6006->6014)

#### weight file 다운로드
1. Super Resolution: ```bash SuperResolution/download-weights.sh```
2. Deblur: ```bash Deblur/download-weights.sh```
