# AI 사진 복원 도구 : Good Bye 옥에티

## 프로젝트 소개
여러분은 평소 사진을 찍을 때 이런 경험 있으신가요? '이 물체만 지우면 완벽한 사진이 될 것 같을 때', '사진의 화질이 좋지 않을 때', '사진이 흐리게 찍혔을 때'. 이럴 때 포토샵이 먼저 떠오르시겠지만, 저희는 포토샵이 너무 어렵고 복잡해 일반인이 사용하기에 큰 부담이 있다고 판단했습니다. 사용자를 대상으로 한 설문 조사 결과, 사용자가 가장 자주 사용하는 ‘크롭’ 기능과, 평소 사용하지 못했던 기능 중 필요하다는 응답이 있었던 ‘특정 물체 지우기’, ‘화질 개선’, ‘흔들림 제거’ 기능을 서비스 하고자 합니다.
![image](https://user-images.githubusercontent.com/40880346/147333892-d9626e0a-a442-48b4-abfb-b106f95aedc4.png)


### 시연 영상
[![Video Label](http://img.youtube.com/vi/Mnqi91GWhiY/0.jpg)](https://www.youtube.com/watch?v=Mnqi91GWhiY)

### 발표 영상
[![Video Label](http://img.youtube.com/vi/wU9lCHz9TI4/0.jpg)](https://www.youtube.com/watch?v=wU9lCHz9TI4)

### 서비스 사용해보기
App : [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/intelli8786/ai_blemishesremover/main/WebServer/Service.py)


## 팀원 소개

||이름|역할|github|
|--|------|---|---|
|<img src=https://user-images.githubusercontent.com/44287798/147333059-cfe32b6a-bef7-45a9-a778-abe425028bf0.png width=100>|김범수|Prototype 개발, SuperResolution 모듈 개발, Github actions|https://github.com/HYU-kbs|
|<img src=https://user-images.githubusercontent.com/44287798/147333099-db64d0ed-bd58-49c3-8453-1eba72194d18.png width=100>|김준태|Prototype 개발, Deblur 모듈 개발, 서비스 고도화 연구|https://github.com/sronger|
|<img src=https://user-images.githubusercontent.com/44287798/147333157-ec9a97c7-b447-4052-917e-97189f3c8615.png width=100>|김지성|PM, Prototype 개발, Inpainting모듈 개발, 학습, Segmentation모듈 개발, REST API 개발, WebAPP 개발, 프로젝트 통합|https://github.com/intelli8786|
|<img src=https://user-images.githubusercontent.com/44287798/147333178-b167a3bc-0d60-4cd3-891d-7ebbddc80a7b.png width=100>|정소희|서비스 요구사항 분석, Prototype 개발, REST API 개발, SuperResolution, 모듈 개발, Error Handling|https://github.com/SoheeJeong|
|<img src=https://user-images.githubusercontent.com/44287798/147333196-579afb0d-0a51-4f87-bcb3-6c78883c1428.png width=100>|홍지연|Prototype 개발, Segmentation 모듈개발, Cloud SQL 및, Storage 연동|https://github.com/hongjourney|



## 서비스 파이프라인
![image](https://user-images.githubusercontent.com/44287798/147332789-174092c5-00e0-43e7-a21e-052020d4955c.png)


## Environment setup
1. Deblur: ```pip install -r Deblur/requirements.txt```

2. Inpainting: ```pip install -r Inpainting/requirements.txt```

3. Segmentation: ```pip install -r Segmentation/requirements.txt```

4. SuperResolution: ```pip install -r SuperResolution/requirements.txt```

5. Streamlit Cloud: ```pip install -r packages.txt```


## 실행방법

#### 서비스 실행 방법
##### Web Server Run
```streamlit run WebServer/Server_SRD.py```
##### Inpainting REST API Server Run
```cd Inpainting```

```python3 Server.py```

##### Semantic REST API Segmentation Server Run
```cd Segmentation```

```python3 Server.py```

##### Super REST API Resolution Server Run
```cd SuperResolution```

```python3 Server.py```

##### Deblur REST API Server Run
```cd Deblur```

```python3 Server.py```

#### pretrained weight file 다운로드
#### Inference code & Pretrained weight Repo
* [SwinIR](https://github.com/JingyunLiang/SwinIR)
* [Deblur](https://github.com/swz30/MPRNet.git)

1. Super Resolution: ```bash SuperResolution/download-weights.sh```
2. Deblur: ```bash Deblur/download-weights.sh```


