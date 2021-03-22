# 시뮬레이터 자율주행

광운대학교 로봇학부 학술소모임 **'BARAM'** 20년도 전반기 LaneDetection Project에 대한 소스코드입니다.  

## 개발 환경
|OS|사용 언어|사용 IDE|
|:---:|:---:|:---:|
|Window 10|C++|Visual Studio Code|

## 프로젝트 개발 동기

-  국제 대학생 창작차 경진대회에 출전했는데 예선전에서 탈락하게 되었다. 오래 준비한 만큼 아쉬움이 남아 내년에 다시 출전했을 때 도움이 되기 위한 기반을 다지기위해 구매한 시뮬레이터를 활용하여 일정수준까지 구현해보고자 다음 작품을 구상하였다. 대회를 참여하기 위해 찾아보았던 이론적인 내용들을 직접 구현해보고 실현이 가능한지 확인해보고 더 나은 방법을 모색하고자 한다. 

## 프로젝트 개요
1. Simulator와 Python 파일 간 UDP통신 구현  
> - IMG 수신
> - 목표 속도, 방향 값 송신
2. Lane detection  
> - 차선의 커브 방향, 곡률 반지름 , 이탈률 도출


## System Architecture
<p align="center"><img src="https://user-images.githubusercontent.com/56825894/100778521-74660780-344a-11eb-993c-ed027cbd7a0e.PNG" width="600px"></p>  


### Project contents

1. 이미지 왜곡 보정
2. ROI설정
3. SOBEL연산 , HSL색공간 사용
4. Perspective transform -> Bird Eye's View  
5. Sliding Window Search 차선위치 탐색  
6. 차선의 이탈율 구함  
7. 차선의 제어량(vel, brake, accel)송신


## 프로젝트 결과

<p align="center"><img src="https://user-images.githubusercontent.com/56825894/100778536-7b8d1580-344a-11eb-91fc-9ceacb25f8e9.gif" width="500px"></p>  
<p align="center"> 직진하는 영상 </p>  
