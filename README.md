CV 14조 final project

## Super Resolution and Deblur module

### Inference code & Pretrained weight Repo
* [SwinIR](https://github.com/JingyunLiang/SwinIR)
* [Deblur](https://github.com/jiangsutx/SRN-Deblur)

### 실행 방법
1. Fast API 사용할 경우 ```python router.py``` 실행
2. Fast API 사용하지 않을 경우 [app.py](https://github.com/boostcampaitech2/final-project-level3-cv-14/blob/22c9ccf8cf85458823a0f9f46660b02a07b45501/src/app.py) 에서 'Fast API 추론' 부분을 주석처리하고 '기존 함수 추론' 부분을 주석해제.
3. ```python main.py --port=6006``` 실행 (열린 port번호 지정) 
4. 실행 후 나타나는 링크 복사
5. 링크에서 port만 변경해서 접속 (필요시에만 변경)(ex:6006->6014)

### Super Resolution weight file 다운로드 방법
> SuperRes/weights/ 폴더와  Deblur/weights/color,gray,lstm/ 폴더에 weight를 다운로드합니다.
1. terminal에서 SuperResolution 폴더 위치로 이동합니다.
2. ```bash download-weights.sh``` 를 실행해 weight를 다운로드합니다.
