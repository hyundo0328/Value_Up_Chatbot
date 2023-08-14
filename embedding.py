<<<<<<< HEAD
from dotenv import load_dotenv
import os
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import matplotlib
import json

load_dotenv()
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
# https://github.com/openai/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb

openai.organization = "org-mjX3YcBmDZokW5pPtX99pTsG"
api_key = os.getenv("api_key")
openai.api_key = api_key




with open("policy.json", "r", encoding='UTF8') as f :
    policy_data = json.load(f)

f.close()

''' 정책에 대한 Embedding '''
policy_data_df = pd.DataFrame(policy_data)
# Embedding을 위한 Column 결합
policy_data_df['combined'] = policy_data_df.apply(lambda row: f"{row['title']}, {row['who']}, {row['when']}", axis=1)
policy_data_df['text_embedding'] = policy_data_df.combined.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
# print(policy_data_df)




with open("customer.json", "r", encoding='UTF8') as f :
    customer_data =  json.load(f)

f.close()
''' 유저에 대한 Embedding '''
customer_data_df = pd.DataFrame(customer_data)
customer_data_df['combined'] = customer_data_df.apply(lambda row: f"{row['title']}, {row['who']}, {row['when']}", axis=1)
customer_data_df['text_embedding'] = customer_data_df.combined.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
# print(customer_data_df)



while(True):
    ''' 상위 데이터 출력 '''
    customer_input = input()

    if(customer_input == "exit"):
        break

    response = openai.Embedding.create(
        input = customer_input,
        model = "text-embedding-ada-002"
    )

    embeddings_customer_question = response['data'][0]['embedding']
    # print(embeddings_customer_question)

    customer_data_df['search_study_history'] = customer_data_df.text_embedding.apply(lambda x: cosine_similarity(x, embeddings_customer_question))
    customer_data_df = customer_data_df.sort_values('search_study_history', ascending=False)
    # print(customer_data_df)

    policy_data_df['search_policy'] = policy_data_df.text_embedding.apply(lambda x: cosine_similarity(x, embeddings_customer_question))
    policy_data_df = policy_data_df.sort_values('search_policy', ascending=False)
    # print(policy_data_df)
    top_3_policy_df = policy_data_df.head(3)
    # print(top_3_policy_df)
    top_3_recommend_df = customer_data_df.head(3)
    # print(top_3_recommend_df)





    ''' 프롬프트 작성 '''
    message_objects = []
    message_objects.append({ "role" : "system", "content" : "너는 정책 추천을 추천해주는 챗봇이야" }) # 시스템
    message_objects.append({ "role":"user", "content": customer_input }) # 유저
    prev_policy = ", ".join([f"{row['combined']}" for index, row in top_3_policy_df.iterrows()])
    message_objects.append({"role":"user", "content": f"내가 관심있는 정책 : {prev_policy}"})
    # 좋은 답변을 위한 프롬프트
    message_objects.append({"role":"user", "content": "너가 추천해준 정책을 자세하게 알려줘"})
    message_objects.append({"role":"user", "content": "추천 정책을 리스트로 정리해서 말해줘"})
    # 챗봇의 답변 형태를 지정해주는 프롬프트
    message_objects.append({"role": "assistant", "content": "제가 3가지 정책을 추천 해주겠습니다."})




    policy_list = []

    for index, row in top_3_policy_df.iterrows():
        policy_dict = {'role': "assistant", "content": f"{row['combined']}"}
        policy_list.append(policy_dict)
    # print(policy_list)

    message_objects.extend(policy_list)
    message_objects.append({"role": "assistant", "content":"당신에게 적합할 것 같은 저의 정책 리스트입니다."})
    # print(message_objects)



    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=message_objects
    )

    print("정책 추천 : ")
    print(completion.choices[0].message['content'])
=======
from dotenv import load_dotenv
import os
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import matplotlib
import json

load_dotenv()
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
# https://github.com/openai/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb

openai.organization = "org-mjX3YcBmDZokW5pPtX99pTsG"
api_key = os.getenv("api_key")
openai.api_key = api_key




with open("policy.json", "r", encoding='UTF8') as f :
    policy_data = json.load(f)

f.close()

''' 정책에 대한 Embedding '''
policy_data_df = pd.DataFrame(policy_data)
# Embedding을 위한 Column 결합
policy_data_df['combined'] = policy_data_df.apply(lambda row: f"{row['title']}, {row['who']}, {row['when']}", axis=1)
policy_data_df['text_embedding'] = policy_data_df.combined.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
# print(policy_data_df)




with open("customer.json", "r", encoding='UTF8') as f :
    customer_data =  json.load(f)

f.close()
''' 유저에 대한 Embedding '''
customer_data_df = pd.DataFrame(customer_data)
customer_data_df['combined'] = customer_data_df.apply(lambda row: f"{row['title']}, {row['who']}, {row['when']}", axis=1)
customer_data_df['text_embedding'] = customer_data_df.combined.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
# print(customer_data_df)



while(True):
    ''' 상위 데이터 출력 '''
    customer_input = input()

    if(customer_input == "exit"):
        break

    response = openai.Embedding.create(
        input = customer_input,
        model = "text-embedding-ada-002"
    )

    embeddings_customer_question = response['data'][0]['embedding']
    # print(embeddings_customer_question)

    customer_data_df['search_study_history'] = customer_data_df.text_embedding.apply(lambda x: cosine_similarity(x, embeddings_customer_question))
    customer_data_df = customer_data_df.sort_values('search_study_history', ascending=False)
    # print(customer_data_df)

    policy_data_df['search_policy'] = policy_data_df.text_embedding.apply(lambda x: cosine_similarity(x, embeddings_customer_question))
    policy_data_df = policy_data_df.sort_values('search_policy', ascending=False)
    # print(policy_data_df)
    top_3_policy_df = policy_data_df.head(3)
    # print(top_3_policy_df)
    top_3_recommend_df = customer_data_df.head(3)
    # print(top_3_recommend_df)





    ''' 프롬프트 작성 '''
    message_objects = []
    message_objects.append({ "role" : "system", "content" : "너는 정책 추천을 추천해주는 챗봇이야" }) # 시스템
    message_objects.append({ "role":"user", "content": customer_input }) # 유저
    prev_policy = ", ".join([f"{row['combined']}" for index, row in top_3_policy_df.iterrows()])
    message_objects.append({"role":"user", "content": f"내가 관심있는 정책 : {prev_policy}"})
    # 좋은 답변을 위한 프롬프트
    message_objects.append({"role":"user", "content": "너가 추천해준 정책을 자세하게 알려줘"})
    message_objects.append({"role":"user", "content": "추천 정책을 리스트로 정리해서 말해줘"})
    # 챗봇의 답변 형태를 지정해주는 프롬프트
    message_objects.append({"role": "assistant", "content": "제가 3가지 정책을 추천 해주겠습니다."})




    policy_list = []

    for index, row in top_3_policy_df.iterrows():
        policy_dict = {'role': "assistant", "content": f"{row['combined']}"}
        policy_list.append(policy_dict)
    # print(policy_list)

    message_objects.extend(policy_list)
    message_objects.append({"role": "assistant", "content":"당신에게 적합할 것 같은 저의 정책 리스트입니다."})
    # print(message_objects)



    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=message_objects
    )

    print("정책 추천 : ")
    print(completion.choices[0].message['content'])
>>>>>>> edde95087f9e282f49b063a9c39256711cfabe31
    print()