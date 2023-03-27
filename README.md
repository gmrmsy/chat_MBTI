# Chat_MBTI
대화형 챗봇을 활용하여 사용자가 작성한 대화 문장을 분석해 사용자의 MBTI를 판별한다.

# 목차
<!-- TOC -->

- [프로젝트 소개](#Chat_MBTI)
- [목차](#목차)
- [

<!-- /TOC -->

# 기존 MBTI검사
기존의 MBTI검사는 상황이나 질문을 제시한 뒤 선택지를 주어 사용자가 선택한 답안을 조합하여 MBTI를 판별하였다.
![설문예시](https://user-images.githubusercontent.com/91594005/227849444-084fb7eb-fd6c-49ef-94ac-b21a49c38da7.png)
이런 경우 질문에서 의문이 생기는 경우, 혹은 선택지 안에서 답을 찾기 어려운 경우 검사진행에 어려움이 있다.

때문에 선택지를 골라 검사를 진행하는 것이 아닌 사용자가 자유롭게 답변을 작성하여 보다 사용자의 생각에 가까운 답변을 제출할 수 있도록 하고자 한다.

이를 구현하기 위해 다음과 같은 과정을 예상해보았다.

1. 챗봇 시스템
2. MBTI 유형별 대화셋
3. 


# 데이터셋
![데이터 출처](https://user-images.githubusercontent.com/91594005/227862619-481d9ba0-239b-43e4-821d-03f95ecc0cbb.png)
AI hub에서 제공하는 '주제별 텍스트 일상 대화 데이터'를 사용한다.


제공하는 말뭉치 안에는 문장의 의도가 라벨링되어있다.
