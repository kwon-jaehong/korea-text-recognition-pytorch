# korea-text-recognition-pytorch

한글 OCR을 만들기 위한 세부 프로젝트
- 현재 제대로 작동 할 리 없음 -

***

작업 순서 
1.학습 문자열 수집 & 정제
2.text recognition 검증 데이터 생성
3.학습


NIA 한국어 형태소 사전
https://kbig.kr/portal/kbig/knowledge/files/bigdata_report.page?bltnNo=10000000016451
"NIADic.xlsx"



필요 모듈

1. 데이터 정제 & 생성
~~글자 길이 스플릿 모듈 (생성)~~
특수문자 폼 생성 하기
딕트 분포를 어느정도 조절하는 모듈

2. 데이터셋 부분
트랜스 포머에
이미지 왜곡,
배경 
글자색

3. 특수문자 포함해서 데이터셋 만들기
4. 데이터 토큰을 유니코드로
5. 디스트리뷰선 병렬처리하기
-> 그냥버젼 , 병렬처리 버젼

6.텐서 보드 연동
7.