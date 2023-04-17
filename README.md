# Chat_MBTI
대화형 챗봇을 활용하여 사용자가 작성한 대화 문장을 분석해 사용자의 MBTI를 판별합니다.

<img src="https://user-images.githubusercontent.com/91594005/232571512-e0c63cf9-9ba7-4926-8a19-0da04b0fc4c2.png"/>

대화 데이터셋과 MBTI문답 데이터 셋으로 챗봇의 대화를 구성하고
사용자의 문장은 감성분석과 의도분석모델로 예측을 한다.
여기서 감성분석은 MBTI의 질문대한 대답이므로 좋고 싫음 / 옳고 그름 / 동의 비동의 기준의 감성분석 모델이 필요하다.
그리고 의도분석 모델을 통해 사용자의 대화 의도의 분포와 빈도로 성격유형을 예측하는데 사용한다.

# 목차
<!-- TOC -->

- [데이터_수집/전처리](#데이터_수집/전처리)
- [감성분류 모델](#감성분류_모델)
- [의도분류_모델](#의도분류_모델)
- [대화형_챗봇](#대화형_챗봇)
- 

<!-- /TOC -->

# 데이터_수집/전처리
<img src="https://user-images.githubusercontent.com/91594005/227862619-481d9ba0-239b-43e4-821d-03f95ecc0cbb.png" width="800" height="300"/>

AI hub에서 제공하는 '주제별 텍스트 일상 대화 데이터'를 사용했습니다.
제공받은 말뭉치 안에는 2,3명의 사람이 주고받은 대화 내용이 들어있습니다.

<img src="https://user-images.githubusercontent.com/91594005/232556745-806877ff-72f9-458d-b7c9-048c62c4755b.png" width="700" height="500"/>

위 그림과 같은 원천데이터와 원천데이터를 json파일로 정리한 라벨링데이터로 구분되어있습니다.

라벨링데이터에는 대화의 상세 정보와 문장 정보가 들어있고, 그 안에 각 문장의 의도가 포함되어 있습니다.
라벨링데이터 부분내용과 의도분류리스트는 다음과 같습니다.

<img src="https://user-images.githubusercontent.com/91594005/228100234-084cf22a-fc24-47b1-b9c0-2fc4cf7f0d64.png" width="800" height="200"/>
<img src="https://user-images.githubusercontent.com/91594005/228100732-bb7cc13d-4f63-4501-b083-fa3d80102dd4.png" width="800" height="100"/>


먼저는 제공받은 라벨링데이터를 한 폴더에 압축을 푼 후 json 라이브러리를 사용해 불러옵니다.

"makedataset/data_collection.ipynb"

```python
import os, json, copy
import pandas as pd
from tqdm import tqdm

path_2 = '주제별 텍스트 일상 대화 데이터/'
file_list_2 = os.listdir(path_2)


li = []
for i in tqdm(range(len(file_list_2))):
    folder = file_list_2[i]

    path_1 = '주제별 텍스트 일상 대화 데이터/'+folder+'/'
    file_list_1 = os.listdir(path_1)

    for n in range(len(file_list_1)):
        with open(path_1+file_list_1[n], 'r', encoding='utf-8') as file:
            temp_data = json.load(file)
            li.append(temp_data)
```


예시 원천데이터처럼 문장에 id가 매칭되지 않은 경우 그 앞 문장과 합쳐주는 함수를 만들었습니다. 

```python
def preprocessing_list(dial_list):
    temp_list = []; cnt = 0
    temp_list = copy.deepcopy(dial_list)
    for i in range(len(dial_list)):
        i -= cnt
        if temp_list[i][0] not in ['1','2','3','4','5'] :
            temp_list[i-1] += temp_list[i].lstrip()
            temp_list.pop(i); cnt += 1
    return temp_list
```


이제 대화데이터셋을 추출합니다.

앞에 만든 함수를 사용하여 원천데이터를 정제하고, 각각 사용자가 발화 한 후 다시 본인의 발화가 나올때까지의 발화를 모두 선발화와 대답으로 쌍을 이루게 했습니다.

그리고 예기치 못한 에러를 방지하기 위해 for문 안에 try/except 문을 사용하여 대화데이터셋을 추출하였습니다.

```python
dialog = pd.DataFrame(columns=['Q','Q_intent', 'A', 'A_intent'])

for i in tqdm(range(len(li))):
    sen = preprocessing_list(li[i])
    intent = li[i]['info'][0]['annotations']['lines']
    for j in range(len(sen)):
        if len(sen) != len(intent):
            print(f"len(sen) != len(intent) : {li[i]['dataset']['name']}")
            break
        else :
            for k in range(1,6):
                if j+k >= len(sen)-1:
                    break
                elif sen[j][0] == sen[j+k][0]:
                    break

                if intent[j]['speechAct'] != intent[j+k]['speechAct']:
                    # dialog = pd.concat([dialog,pd.DataFrame({'Q' : [sen[j].split(':')[-1].lstrip()], 'Q_intent' : intent[j]['speechAct'], 'A' : [sen[j+k].split(':')[-1].lstrip()], 'A_intent' : intent[j+k]['speechAct']})],ignore_index=True)
                    try:
                        dialog = pd.concat([dialog,pd.DataFrame({'Q' : [sen[j].split(':')[-1].lstrip()],
                                                                 'Q_intent' : intent[j]['speechAct'],
                                                                 'A' : [sen[j+k].split(':')[-1].lstrip()],
                                                                 'A_intent' : intent[j+k]['speechAct']})], ignore_index=True)
                    except :
                        print(li[i]['dataset']['name'])
                        print(f'대화번호i:{i} / 문장번호j:{j} / 쌍번호k:{k}')

```

위 코드를 통해 만들어진 대화데이터셋은 다음과 같습니다.

<img src="https://user-images.githubusercontent.com/91594005/232559955-1bb836dd-8787-41b0-bf5c-22e2412dc78c.png" height=400>



다음으로 의도분류와 감성분류를 위한 데이터셋을 추출합니다.

```python
dialog = pd.read_csv('dataset/dialog_chatbot.csv')

intent_model = pd.DataFrame(columns=['sentence','intent'])
Q_temp_sen = pd.DataFrame()
Q_temp_int = pd.DataFrame()
Q_temp_sen['sentence'] = dialog['Q']
Q_temp_int['intent'] = dialog['Q_intent']
Q_temp = pd.concat([Q_temp_sen,Q_temp_int], axis=1)

A_temp_sen = pd.DataFrame()
A_temp_int = pd.DataFrame()
A_temp_sen['sentence'] = dialog['A']
A_temp_int['intent'] = dialog['A_intent']
A_temp = pd.concat([A_temp_sen,A_temp_int], axis=1)

intent_model = pd.concat([Q_temp,A_temp], ignore_index=True)
intent_model.drop(intent_model.loc[intent_model['intent']=='N/A'].index, axis=0, inplace=True)
intent_model

intent_model.drop(['Unnamed: 0'], axis=1, inplace=True)    # 불필요 column 제거
intent_model.dropna(inplace=True)    # 결측치 제거
intent_model.drop(intent_model[intent_model.duplicated()].index, axis=0, inplace=True)    # 중복값 제거
intent_model.drop(intent_model[intent_model['intent'] == '(언약) 위협하기'].index, axis=0, inplace=True)    # 불필요 label 제거
intent_model.reset_index(drop=True, inplace=True)    # 인덱스 초기화

intent_model.to_csv('intent_model_dataset.csv')
```

위 코드를 통해 만들어진 의도/감성분류 데이터셋은 다음과 같습니다.

<img src="https://user-images.githubusercontent.com/91594005/232564995-87a38165-e044-4023-bf24-3fdba7c7b936.png" height=400/>


# 감성분류_모델

<img src="https://user-images.githubusercontent.com/91594005/232574582-2e2130c1-769f-4f43-9291-5ca152bec5fb.png"/>






<img src="" width="800" height="200"/>
