from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# 1. 형태소 분석 및 명사, 동사 추출
def extract_keywords_with_kiwi(texts):
    kiwi = Kiwi()
    processed_texts = []
    
    for text in texts:
        tokens = kiwi.analyze(text)[0][0]
        filtered_tokens = [
            token.form
            for token in tokens
            # NN 명사 (NNG-일반명사 ,NNP-고유명사, NNB-의존명사) VV 동사 SL 영어 
            #명사,동사,영어인것만 filtered_tokens에 넣기기
            if token.tag.startswith("NN") or token.tag.startswith("VV") or token.tag.startswith('SL')
        ]
        processed_texts.append(" ".join(filtered_tokens))
    
    return processed_texts

# 2. TF-IDF를 이용한 핵심 키워드 추출
# top_n 추출할 키워드 개수
def extract_keywords_tfidf(processed_texts, top_n=3):
    # python에서 사용할 경우에 단순 빈도(TF)는 Counvectorizer 클래스를, TF IDF의 경우 TfidVectorizer클래스를 사용한다. 
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    #피쳐 이름 출력력
    feature_names = vectorizer.get_feature_names_out()
    
    # 문서별로 TF-IDF 상위 키워드 추출
    keywords_per_doc = []
    for row in tfidf_matrix:
        scores = row.toarray()[0]
        #내림차순 정렬하고 5개까지 가져오기
        top_indices = np.argsort(scores)[::-1][:top_n]
        keywords = [(feature_names[i], scores[i]) for i in top_indices]
        keywords_per_doc.append(keywords)
    
    return keywords_per_doc


    
if __name__ == "__main__":

# 기사 텍스트 샘플 (한글 기사)
    articles = [
        "대한민국 대표 보컬리스트 김범수, 박정현의 로맨틱 콜라보레이션 [하얀 겨울]\n\n11월 30일(금) 그들의 달콤한 겨울 이야기가 시작된다.\n\n매년 추운 겨울이 되면 어김없이 생각나는 노래, 지난 1993년 발표 이후 지금까지 겨울 음악의 스테디 셀러라 해도 과언이 아닐 만큼 많은 사람들에게 사랑 받아 오고 있는 Mr.2(미스터투)의 [하얀 겨울]이 대한민국 대표 보컬리스트 김범수, 박정현에 의해 새롭게 태어났다",
        "겨울이라 넘 춥지만 행복해요~ 따뜻한 아메리카노까지 완벽"
        "재미있는 노랫말과 쉽게 따라할 수 있는 손동작으로 겨울 동요를 배워보아요 눈사람도 만들면서"
        # 'banana apple apple eggplant', 
        # 'orange carrot banana eggplant', 
        # 'apple carrot banana banana'
    ]
    
    # 3. 실행
    processed_texts = extract_keywords_with_kiwi(articles)
    keywords_per_doc = extract_keywords_tfidf(processed_texts)
    
    # 결과 출력
    for i, keywords in enumerate(keywords_per_doc):
        print(f"Document {i + 1} Keywords:")
        for keyword, score in keywords:
            print(f"  {keyword}: {score:.4f}")
        print()

#결과값 
# Document 1 Keywords:
#   겨울: 0.4045
#   보컬: 0.2739
#   김범수: 0.2739

# Document 2 Keywords:
#   행복: 0.5465
#   아메리카노: 0.5465
#   완벽: 0.5465

# Document 3 Keywords:
#   따르: 0.3799
#   노랫말: 0.3799
#   만들: 0.3799
