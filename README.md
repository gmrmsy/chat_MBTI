# Chat_MBTI
대화형 챗봇을 활용하여 사용자가 작성한 대화 문장을 분석해 사용자의 MBTI를 판별한다.

# 목차
<!-- TOC -->

- [데이터_수집](#데이터_수집)
- [데이터_처리](#데이터_전처리)
- [의도분류_모델](#의도분류_모델)
- [감성분류 모델](#감성분류_모델)
- [대화형_챗봇](#대화형_챗봇)
- 

<!-- /TOC -->

# 데이터_수집
<img src="https://user-images.githubusercontent.com/91594005/227862619-481d9ba0-239b-43e4-821d-03f95ecc0cbb.png" width="800" height="300"/>

AI hub에서 제공하는 '주제별 텍스트 일상 대화 데이터'를 사용했다.
제공받은 말뭉치 안에는 2,3명의 사람이 주고받은 대화 내용이 들어있다.

<img src="https://user-images.githubusercontent.com/91594005/228107409-7fc451d9-5ffc-4fa8-ab21-bf7d0b3e9c72.png" width="450" height="500"/>

위 그림과 같은 원천데이터와 원천데이터를 json파일로 정리한 라벨링데이터로 구분되어져있다.

라벨링데이터에는 대화의 상세 정보와 문장 정보가 들어있고, 그 안에 각 문장의 의도가 포함되어 있다.
라벨링데이터 부분내용과 의도분류리스트는 다음과 같다.

<img src="https://user-images.githubusercontent.com/91594005/228100234-084cf22a-fc24-47b1-b9c0-2fc4cf7f0d64.png" width="800" height="200"/>
<img src="https://user-images.githubusercontent.com/91594005/228100732-bb7cc13d-4f63-4501-b083-fa3d80102dd4.png" width="800" height="100"/>

먼저는 단일 문장에 대한 의도를 분류하기 위해서 각각의 문장과 문장의 의도만 추출하며 데이터셋을 만들었다.

```python
import os, json
import pandas as pd

path_2 = '주제별 텍스트 일상 대화 데이터/'
file_list_2 = os.listdir(path_2)
print(file_list_2)

intent = pd.DataFrame(columns=['sentence', 'intent'])

li = []
for folder in file_list_2:

    print(folder)
    path_1 = '주제별 텍스트 일상 대화 데이터/'+folder+'/'
    file_list_1 = os.listdir(path_1)
    for n in range(len(file_list_1)):
        with open(path_1+file_list_1[n], 'r', encoding='utf-8') as file:
            temp_data = json.load(file)
            
            for line in temp_data['info'][0]['annotations']['lines']:
                
                dialog = pd.concat([dialog,pd.DataFrame({'sentence' : [line['norm_text']], 'intent' : [line['speechAct']]})],ignore_index=True)

print(intent)
```

그 다음결측값과 중복값을 제거해준다.
그리고 문장안에 개인정보가 들어있는 경우 '\*' 로 표기를 했기 때문에 '\*'을 포함한 문장 또한 제거해준다.

```ptyhon
intent.dropna(inplace=True)    # 결측치 제거
intent.drop(df[df.duplicated()].index, axis=0, inplace=True)    # 중복값 제거
intent.drop(df.loc[df['sentence'].str.contains('\*')].index, axis=0, inplace=True)    # '*' 포함한 문장 제거
intent_drop(intent.loc[intent['intent']=='N/A'].index, axis=0, inplace=True)    # intent열 'N/A'값 제거
intent.reset_index(drop=True, inplace=True)    # 인덱스 초기화
```

<img src="https://user-images.githubusercontent.com/91594005/228116685-4b739a5d-b9d0-4254-a51d-7c60428ea0d5.png" width="600" height="300"/>






<img src="" width="800" height="200"/>
