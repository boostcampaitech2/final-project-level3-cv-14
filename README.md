CV 14조 final project
# AI 사진 복원 도구 : Good Bye 옥에티

## 프로젝트 소개

### 시연 영상

[![Video Label](http://img.youtube.com/vi/Mnqi91GWhiY/0.jpg)](https://www.youtube.com/watch?v=Mnqi91GWhiY)

### 서비스 사용해보기
App : [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/intelli8786/ai_blemishesremover/main/WebServer/Service.py)


## 팀원 소개

||이름|역할|github|
|--|------|---|---|
|<img src=https://user-images.githubusercontent.com/44287798/147333059-cfe32b6a-bef7-45a9-a778-abe425028bf0.png width=100>|김범수|Prototype 개발,SuperResolution 모듈 개발,Github actions|https://github.com/HYU-kbs|
|<img src=https://user-images.githubusercontent.com/44287798/147333099-db64d0ed-bd58-49c3-8453-1eba72194d18.png width=100>|김준태|Prototype 개발,Deblur 모듈 개발,서비스 고도화 연구|https://github.com/sronger|
|<img src=https://user-images.githubusercontent.com/44287798/147333157-ec9a97c7-b447-4052-917e-97189f3c8615.png width=100>|김지성|PM,Prototype 개발,Inpainting모듈 개발,학습,Segmentation모듈 개발,REST API 개발,WebAPP 개발,프로젝트 통합|https://github.com/intelli8786|
|<img src=https://user-images.githubusercontent.com/44287798/147333178-b167a3bc-0d60-4cd3-891d-7ebbddc80a7b.png width=100>|정소희|서비스 요구사항 분석,Prototype 개발,REST API 개발,SuperResolution,모듈 개발,Error Handling|https://github.com/SoheeJeong|
|<img src=https://user-images.githubusercontent.com/44287798/147333196-579afb0d-0a51-4f87-bcb3-6c78883c1428.png width=100>|홍지연|Prototype 개발,Segmentation 모듈개발,Cloud SQL 및,Storage 연동|https://github.com/hongjourney|



## 서비스 파이프라인
![image](https://user-images.githubusercontent.com/44287798/147332789-174092c5-00e0-43e7-a21e-052020d4955c.png)


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
