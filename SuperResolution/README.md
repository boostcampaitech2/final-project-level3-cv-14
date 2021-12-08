## Super Resolution and Deblur module

### Inference code & Pretrained weight Repo
* [SwinIR](https://github.com/JingyunLiang/SwinIR)
* [Deblur](https://github.com/jiangsutx/SRN-Deblur)

### 실행 방법
1. ```python main.py --port=6006``` 실행 (열린 port번호 지정) 
2. 실행 후 나타나는 링크 복사
3. 링크에서 port만 변경해서 접속 (필요시에만 변경)(ex:6006->6014)

### Super Resolution weight file 다운로드 방법
> 각 모듈별로 SuperRes/weights/ 폴더에 weight를 다운로드합니다.
1. terminal에서 ```cd SuperResolution``` 으로 SuperResolution 폴더 위치로 이동합니다.
2. terminal 에서 ```bash download-weights.sh``` 를 실행해 weight를 다운로드합니다.
